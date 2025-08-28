import time
import requests
from typing import Dict, Any, Optional
import sys
import os
from ..models.card import Card
from ..models.search import SearchQuery, SearchResult
from ..config import SCRYFALL_BASE_URL, SCRYFALL_RATE_LIMIT_MS, MAX_RESULTS_PER_SEARCH, ENABLE_FULL_PAGINATION, MAX_PAGES_TO_FETCH

# Import event system if available
try:
    from ..events import SearchEventType, create_scryfall_pagination_started_event, create_scryfall_page_fetched_event, create_scryfall_pagination_completed_event, create_error_event
except ImportError:
    # Graceful fallback if events module is not available
    SearchEventType = None


class ScryfallAPI:
    """Tool for interacting with the Scryfall API"""
    
    def __init__(self, event_emitter=None):
        self.base_url = SCRYFALL_BASE_URL
        self.rate_limit_ms = SCRYFALL_RATE_LIMIT_MS
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTGSearchAgent/1.0',
            'Accept': 'application/json'
        })
        self.events = event_emitter
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        current_time = time.time() * 1000  # Convert to milliseconds
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_ms:
            sleep_time = (self.rate_limit_ms - time_since_last) / 1000  # Convert back to seconds
            time.sleep(sleep_time)
        
        self.last_request_time = time.time() * 1000
    
    def search_cards(self, search_query: SearchQuery) -> SearchResult:
        """
        Search for cards using Scryfall API with optional pagination support
        
        Args:
            search_query: The query to search for
            
        Returns:
            SearchResult containing matching cards (potentially across multiple pages)
        """
        if ENABLE_FULL_PAGINATION:
            return self._search_cards_paginated(search_query)
        else:
            return self._search_cards_single_page(search_query)
    
    def _search_cards_single_page(self, search_query: SearchQuery) -> SearchResult:
        """
        Search for cards using single page (original behavior)
        """
        self._rate_limit()
        
        params = {
            'q': search_query.query,
            'unique': 'cards',
            'order': 'name',
            'page': 1
        }
        
        try:
            response = self.session.get(f"{self.base_url}/cards/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert API response to our models
            cards = []
            for card_data in data.get('data', [])[:MAX_RESULTS_PER_SEARCH]:
                cards.append(Card.from_scryfall(card_data))
            
            return SearchResult(
                query=search_query,
                cards=cards,
                total_cards=data.get('total_cards', 0),
                has_more=data.get('has_more', False)
            )
            
        except requests.exceptions.RequestException as e:
            # Return empty result on error
            return SearchResult(
                query=search_query,
                cards=[],
                total_cards=0,
                has_more=False
            )
    
    def _search_cards_paginated(self, search_query: SearchQuery) -> SearchResult:
        """
        Search for cards with full pagination support to get complete result sets
        """
        all_cards = []
        total_cards = 0
        has_more = True
        page = 1
        pages_fetched = 0
        
        # Emit pagination started event
        if self.events and SearchEventType:
            self.events.emit(SearchEventType.SCRYFALL_PAGINATION_STARTED, 
                           create_scryfall_pagination_started_event(search_query.query))
        
        while has_more and pages_fetched < MAX_PAGES_TO_FETCH:
            self._rate_limit()
            
            params = {
                'q': search_query.query,
                'unique': 'cards',
                'order': 'name',
                'page': page
            }
            
            try:
                response = self.session.get(f"{self.base_url}/cards/search", params=params)
                response.raise_for_status()
                data = response.json()
                
                # Convert API response to our models
                page_cards = []
                for card_data in data.get('data', []):
                    page_cards.append(Card.from_scryfall(card_data))
                
                all_cards.extend(page_cards)
                total_cards = data.get('total_cards', 0)
                has_more = data.get('has_more', False)
                pages_fetched += 1
                
                # Emit page fetched event
                if self.events and SearchEventType:
                    self.events.emit(SearchEventType.SCRYFALL_PAGE_FETCHED,
                                   create_scryfall_page_fetched_event(page, len(page_cards), len(all_cards), total_cards))
                
                # Respect MAX_RESULTS_PER_SEARCH limit across all pages
                if len(all_cards) >= MAX_RESULTS_PER_SEARCH:
                    all_cards = all_cards[:MAX_RESULTS_PER_SEARCH]
                    has_more = False
                    # Emit completion event with limit information
                    if self.events and SearchEventType:
                        self.events.emit(SearchEventType.SCRYFALL_PAGINATION_COMPLETED,
                                       create_scryfall_pagination_completed_event(len(all_cards), pages_fetched, True))
                    break
                
                page += 1
                
            except requests.exceptions.RequestException as e:
                # Emit error event
                if self.events and SearchEventType:
                    self.events.emit(SearchEventType.ERROR_OCCURRED,
                                   create_error_event("scryfall_request_error", str(e), {"page": page, "query": search_query.query}))
                
                # If we have some cards already, return them; otherwise return empty
                if all_cards:
                    has_more = False
                    break
                else:
                    return SearchResult(
                        query=search_query,
                        cards=[],
                        total_cards=0,
                        has_more=False
                    )
        
        # Emit final completion event
        if self.events and SearchEventType:
            limited_by_max = pages_fetched >= MAX_PAGES_TO_FETCH and has_more
            self.events.emit(SearchEventType.SCRYFALL_PAGINATION_COMPLETED,
                           create_scryfall_pagination_completed_event(len(all_cards), pages_fetched, limited_by_max))
        
        return SearchResult(
            query=search_query,
            cards=all_cards,
            total_cards=total_cards,
            has_more=has_more and pages_fetched >= MAX_PAGES_TO_FETCH  # Only true if we stopped due to page limit
        )
    
    def get_random_card(self) -> Optional[Card]:
        """Get a random card from Scryfall API for testing"""
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.base_url}/cards/random")
            response.raise_for_status()
            data = response.json()
            return Card.from_scryfall(data)
        except requests.exceptions.RequestException:
            return None