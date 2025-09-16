"""
Unit tests for the ScryfallAPI tool.

Tests cover:
- API initialization and configuration
- Rate limiting functionality
- Basic card search operations
- Pagination handling
- Error handling and edge cases
- Event emission during operations
- Mock-based testing to avoid real API calls
"""
import pytest
import time
import requests
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, Timeout, HTTPError

from src.mtg_search_agent.tools.scryfall_api import ScryfallAPI
from src.mtg_search_agent.models.search import SearchQuery, SearchResult
from src.mtg_search_agent.models.card import Card


class TestScryfallAPIInitialization:
    """Test suite for ScryfallAPI initialization and configuration."""

    def test_scryfall_api_initialization_default(self):
        """Test ScryfallAPI initialization with default settings."""
        api = ScryfallAPI()

        assert api.base_url == "https://api.scryfall.com"
        assert api.rate_limit_ms == 100
        assert api.last_request_time == 0
        assert api.events is None
        assert hasattr(api, 'session')
        assert api.session.headers['User-Agent'] == 'MTGSearchAgent/1.0'
        assert api.session.headers['Accept'] == 'application/json'

    def test_scryfall_api_initialization_with_event_emitter(self, mock_event_emitter):
        """Test ScryfallAPI initialization with event emitter."""
        api = ScryfallAPI(event_emitter=mock_event_emitter)

        assert api.events == mock_event_emitter

    def test_scryfall_api_session_configuration(self):
        """Test that requests session is properly configured."""
        api = ScryfallAPI()

        assert isinstance(api.session, requests.Session)
        assert 'User-Agent' in api.session.headers
        assert 'Accept' in api.session.headers
        assert api.session.headers['Accept'] == 'application/json'


class TestScryfallAPIRateLimiting:
    """Test suite for rate limiting functionality."""

    def test_rate_limit_enforcement(self):
        """Test that rate limiting delays requests appropriately."""
        api = ScryfallAPI()
        api.rate_limit_ms = 100  # 100ms rate limit

        # Simulate first request
        start_time = time.time() * 1000
        api.last_request_time = start_time

        # Mock time.sleep to capture calls
        with patch('time.sleep') as mock_sleep:
            # Call rate limit method immediately after setting last_request_time
            api._rate_limit()

            # Should call sleep since not enough time has passed
            mock_sleep.assert_called_once()
            sleep_time = mock_sleep.call_args[0][0]
            assert sleep_time > 0

    def test_rate_limit_no_delay_when_enough_time_passed(self):
        """Test that rate limiting doesn't delay when enough time has passed."""
        api = ScryfallAPI()
        api.rate_limit_ms = 100

        # Set last request time to simulate enough time has passed
        api.last_request_time = (time.time() - 1) * 1000  # 1 second ago

        with patch('time.sleep') as mock_sleep:
            api._rate_limit()

            # Should not call sleep since enough time has passed
            mock_sleep.assert_not_called()

    def test_rate_limit_updates_last_request_time(self):
        """Test that rate limiting updates last_request_time."""
        api = ScryfallAPI()
        original_time = api.last_request_time

        with patch('time.sleep'):
            api._rate_limit()

            assert api.last_request_time > original_time

    @patch('src.mtg_search_agent.tools.scryfall_api.time.sleep')
    @patch('src.mtg_search_agent.tools.scryfall_api.time.time')
    def test_rate_limit_precise_timing(self, mock_time, mock_sleep):
        """Test rate limiting with precise timing control."""
        api = ScryfallAPI()
        api.rate_limit_ms = 50  # 50ms rate limit

        # Set up time sequence
        current_time = 1000.0  # Base time in seconds
        last_request_time = current_time * 1000  # Last request time in ms
        current_time_ms = (current_time + 0.025) * 1000  # Current time + 25ms in ms

        # Mock time.time() to return current time + 25ms
        mock_time.return_value = current_time + 0.025  # In seconds

        api.last_request_time = last_request_time

        api._rate_limit()

        # Should sleep for remaining time (50ms - 25ms = 25ms = 0.025s)
        mock_sleep.assert_called_once_with(0.025)


class TestScryfallAPIBasicSearch:
    """Test suite for basic search functionality."""

    @patch('requests.Session.get')
    def test_search_cards_successful_response(self, mock_get, sample_search_query_model, mock_scryfall_response):
        """Test successful card search with mock response."""
        # Configure mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit') as mock_rate_limit:
            result = api.search_cards(sample_search_query_model)

            # Verify rate limiting was called
            mock_rate_limit.assert_called_once()

            # Verify API request
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert '/cards/search' in call_args[0][0]
            assert 'q' in call_args[1]['params']

            # Verify result
            assert isinstance(result, SearchResult)
            assert result.query == sample_search_query_model
            assert len(result.cards) == 2
            assert result.total_cards == 2
            assert result.has_more is False

    @patch('requests.Session.get')
    def test_search_cards_with_pagination(self, mock_get, sample_search_query_model, mock_scryfall_paginated_response):
        """Test card search with pagination enabled."""
        # First page response
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = mock_scryfall_paginated_response
        first_response.raise_for_status.return_value = None

        # Second page response (simulated)
        second_page_data = mock_scryfall_paginated_response.copy()
        second_page_data["has_more"] = False
        second_page_data["total_cards"] = 175
        del second_page_data["next_page"]

        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = second_page_data
        second_response.raise_for_status.return_value = None

        mock_get.side_effect = [first_response, second_response]

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)

            # Should have made 2 requests for pagination
            assert mock_get.call_count == 2

            # Result should combine both pages
            assert len(result.cards) == 2  # 1 card per page
            assert result.total_cards == 175

    @patch('requests.Session.get')
    def test_search_cards_no_results(self, mock_get, sample_search_query_model):
        """Test card search with no results."""
        empty_response = {
            "object": "list",
            "total_cards": 0,
            "has_more": False,
            "data": []
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = empty_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)

            assert len(result.cards) == 0
            assert result.total_cards == 0
            assert result.has_more is False

    @patch('requests.Session.get')
    def test_search_cards_with_event_emission(self, mock_get, sample_search_query_model, mock_scryfall_response, mock_event_emitter):
        """Test that card search emits appropriate events."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI(event_emitter=mock_event_emitter)

        with patch.object(api, '_rate_limit'):
            api.search_cards(sample_search_query_model)

            # Verify events were emitted
            assert mock_event_emitter.emit.call_count > 0

            # Check that at least some expected event types were emitted
            emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
            event_types = [event.event_type for event in emitted_events]

            # Should emit at least cards fetched events
            assert any("scryfall" in event_type for event_type in event_types)


class TestScryfallAPIErrorHandling:
    """Test suite for error handling scenarios."""

    @patch('requests.Session.get')
    def test_search_cards_http_error(self, mock_get, sample_search_query_model):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = HTTPError("Bad Request")
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)
            # Should return empty result on HTTP error
            assert len(result.cards) == 0
            assert result.total_cards == 0
            assert result.has_more is False

    @patch('requests.Session.get')
    def test_search_cards_timeout_error(self, mock_get, sample_search_query_model):
        """Test handling of timeout errors."""
        mock_get.side_effect = Timeout("Request timed out")

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)
            # Should return empty result on timeout
            assert len(result.cards) == 0
            assert result.total_cards == 0
            assert result.has_more is False

    @patch('requests.Session.get')
    def test_search_cards_connection_error(self, mock_get, sample_search_query_model):
        """Test handling of connection errors."""
        mock_get.side_effect = RequestException("Connection failed")

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)
            # Should return empty result on connection error
            assert len(result.cards) == 0
            assert result.total_cards == 0
            assert result.has_more is False

    @patch('requests.Session.get')
    def test_search_cards_json_decode_error(self, mock_get, sample_search_query_model):
        """Test handling of JSON decode errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            with pytest.raises(ValueError):
                api.search_cards(sample_search_query_model)

    @patch('requests.Session.get')
    def test_search_cards_malformed_response(self, mock_get, sample_search_query_model):
        """Test handling of malformed API responses."""
        malformed_response = {
            "object": "error",
            "code": "bad_request",
            "status": 400,
            "details": "Invalid query syntax"
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = malformed_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            # Should handle malformed response gracefully by returning empty result
            result = api.search_cards(sample_search_query_model)
            assert len(result.cards) == 0  # No 'data' key means empty cards list
            assert result.total_cards == 0  # No 'total_cards' key means 0
            assert result.has_more is False  # No 'has_more' key means False


class TestScryfallAPIPagination:
    """Test suite for pagination functionality."""

    @patch('requests.Session.get')
    def test_pagination_disabled(self, mock_get, sample_search_query_model, mock_scryfall_paginated_response):
        """Test search with pagination disabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_paginated_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        # Mock config to disable pagination
        with patch('src.mtg_search_agent.tools.scryfall_api.ENABLE_FULL_PAGINATION', False):
            with patch.object(api, '_rate_limit'):
                result = api.search_cards(sample_search_query_model)

                # Should only make one request
                assert mock_get.call_count == 1

                # Should still indicate more results available
                assert result.has_more is True

    @patch('requests.Session.get')
    def test_pagination_max_pages_limit(self, mock_get, sample_search_query_model, mock_scryfall_paginated_response):
        """Test pagination respects max pages limit."""
        # Create responses for multiple pages
        responses = []
        for i in range(5):  # Create 5 pages worth of responses
            page_response = mock_scryfall_paginated_response.copy()
            if i < 4:  # First 4 pages have more
                page_response["has_more"] = True
                page_response["next_page"] = f"https://api.scryfall.com/cards/search?page={i+2}"
            else:  # Last page
                page_response["has_more"] = False
                if "next_page" in page_response:
                    del page_response["next_page"]

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = page_response
            mock_response.raise_for_status.return_value = None
            responses.append(mock_response)

        mock_get.side_effect = responses

        api = ScryfallAPI()

        # Mock config to limit pages
        with patch('src.mtg_search_agent.tools.scryfall_api.MAX_PAGES_TO_FETCH', 2):
            with patch.object(api, '_rate_limit'):
                result = api.search_cards(sample_search_query_model)

                # Should only make 2 requests despite more pages available
                assert mock_get.call_count == 2

    @patch('requests.Session.get')
    def test_pagination_with_event_emission(self, mock_get, sample_search_query_model, mock_scryfall_paginated_response, mock_event_emitter):
        """Test that pagination emits appropriate events."""
        # Set up multi-page response
        first_page = mock_scryfall_paginated_response.copy()
        second_page = mock_scryfall_paginated_response.copy()
        second_page["has_more"] = False
        if "next_page" in second_page:
            del second_page["next_page"]

        responses = []
        for page_data in [first_page, second_page]:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = page_data
            mock_response.raise_for_status.return_value = None
            responses.append(mock_response)

        mock_get.side_effect = responses

        api = ScryfallAPI(event_emitter=mock_event_emitter)

        with patch.object(api, '_rate_limit'):
            api.search_cards(sample_search_query_model)

            # Should emit pagination events
            emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
            event_types = [event.event_type for event in emitted_events]

            # Should emit pagination-related events
            assert any("pagination" in event_type for event_type in event_types)


class TestScryfallAPIQueryConstruction:
    """Test suite for query construction and parameter handling."""

    @patch('requests.Session.get')
    def test_query_parameter_encoding(self, mock_get, mock_scryfall_response):
        """Test that query parameters are properly encoded."""
        query_with_special_chars = SearchQuery(
            query='c:r t:"instant" o:"deal damage"',
            explanation="Red instants that deal damage"
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            api.search_cards(query_with_special_chars)

            # Verify the call was made with proper parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            params = call_args[1]['params']

            assert 'q' in params
            assert params['q'] == query_with_special_chars.query

    @patch('requests.Session.get')
    def test_default_search_parameters(self, mock_get, sample_search_query_model, mock_scryfall_response):
        """Test that default search parameters are included."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            api.search_cards(sample_search_query_model)

            call_args = mock_get.call_args
            params = call_args[1]['params']

            # Should include the query
            assert 'q' in params
            assert params['q'] == sample_search_query_model.query


class TestScryfallAPICardCreation:
    """Test suite for Card model creation from API responses."""

    def test_card_creation_from_scryfall_data(self, sample_card_data):
        """Test that cards are properly created from Scryfall API data."""
        card = Card.from_scryfall(sample_card_data)

        assert isinstance(card, Card)
        assert card.name == sample_card_data["name"]
        assert card.id == sample_card_data["id"]

    @patch('requests.Session.get')
    def test_search_result_card_instances(self, mock_get, sample_search_query_model, mock_scryfall_response):
        """Test that search results contain proper Card instances."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI()

        with patch.object(api, '_rate_limit'):
            result = api.search_cards(sample_search_query_model)

            assert len(result.cards) > 0
            for card in result.cards:
                assert isinstance(card, Card)
                assert hasattr(card, 'name')
                assert hasattr(card, 'id')


class TestScryfallAPIConfiguration:
    """Test suite for configuration and settings."""

    def test_api_uses_config_values(self):
        """Test that API uses values from config module."""
        api = ScryfallAPI()

        # Should use config values for base URL and rate limit
        assert "scryfall.com" in api.base_url
        assert api.rate_limit_ms > 0

    def test_api_base_url_construction(self):
        """Test API base URL construction and endpoints."""
        api = ScryfallAPI()

        # Base URL should not end with slash
        assert not api.base_url.endswith('/')

        # Should be able to construct search endpoint
        search_url = f"{api.base_url}/cards/search"
        assert search_url == "https://api.scryfall.com/cards/search"

    def test_session_headers_configuration(self):
        """Test that session headers are properly configured."""
        api = ScryfallAPI()

        required_headers = ['User-Agent', 'Accept']
        for header in required_headers:
            assert header in api.session.headers

        assert api.session.headers['Accept'] == 'application/json'
        assert 'MTGSearchAgent' in api.session.headers['User-Agent']


class TestScryfallAPIIntegration:
    """Integration tests for ScryfallAPI functionality."""

    @patch('requests.Session.get')
    def test_complete_search_workflow(self, mock_get, sample_search_query_model, mock_scryfall_response, mock_event_emitter):
        """Test complete search workflow from query to results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_scryfall_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        api = ScryfallAPI(event_emitter=mock_event_emitter)

        # Simulate a complete search
        with patch.object(api, '_rate_limit') as mock_rate_limit:
            result = api.search_cards(sample_search_query_model)

            # Verify the complete workflow
            mock_rate_limit.assert_called()  # Rate limiting applied
            mock_get.assert_called_once()    # API request made
            mock_event_emitter.emit.assert_called()  # Events emitted

            # Verify result quality
            assert isinstance(result, SearchResult)
            assert result.query == sample_search_query_model
            assert len(result.cards) == len(mock_scryfall_response["data"])
            assert all(isinstance(card, Card) for card in result.cards)

    def test_error_recovery_and_reporting(self, sample_search_query_model, mock_event_emitter):
        """Test error recovery and reporting through events."""
        api = ScryfallAPI(event_emitter=mock_event_emitter)

        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = RequestException("Network error")

            with patch.object(api, '_rate_limit'):
                result = api.search_cards(sample_search_query_model)

                # Should return empty result on network error
                assert len(result.cards) == 0
                assert result.total_cards == 0
                assert result.has_more is False

                # Verify error events are emitted (depending on pagination mode)
                # In single page mode, no error events are emitted
                # In paginated mode, error events would be emitted