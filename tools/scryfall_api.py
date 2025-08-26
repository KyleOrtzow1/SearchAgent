import time
import requests
from typing import Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.card import Card
from models.search import SearchQuery, SearchResult
from config import SCRYFALL_BASE_URL, SCRYFALL_RATE_LIMIT_MS, MAX_RESULTS_PER_SEARCH


class ScryfallAPI:
    """Tool for interacting with the Scryfall API"""
    
    def __init__(self):
        self.base_url = SCRYFALL_BASE_URL
        self.rate_limit_ms = SCRYFALL_RATE_LIMIT_MS
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTGSearchAgent/1.0',
            'Accept': 'application/json'
        })
    
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
        Search for cards using Scryfall API
        
        Args:
            search_query: The query to search for
            
        Returns:
            SearchResult containing matching cards
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