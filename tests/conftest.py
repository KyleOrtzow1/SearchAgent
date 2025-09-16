"""
Pytest configuration and shared fixtures.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List, Any
from datetime import datetime
import os

# Mock all OpenAI API calls at module level to prevent any real API requests
@pytest.fixture(autouse=True)
def mock_openai_api():
    """Automatically mock all OpenAI API calls for every test"""
    with patch('openai.OpenAI') as mock_openai_client, \
         patch('openai.AsyncOpenAI') as mock_async_openai_client, \
         patch('pydantic_ai.models.openai.OpenAIResponsesModel') as mock_openai_model, \
         patch('src.mtg_search_agent.agents.query_agent.query_agent') as mock_query_agent, \
         patch('src.mtg_search_agent.agents.evaluation_agent.lightweight_evaluation_agent') as mock_eval_agent, \
         patch('src.mtg_search_agent.agents.evaluation_agent.feedback_synthesis_agent') as mock_feedback_agent:

        # Mock OpenAI clients to prevent any direct API calls
        mock_openai_client.return_value = Mock()
        mock_async_openai_client.return_value = Mock()
        mock_openai_model.return_value = Mock()

        # Mock the query agent with realistic return types
        from src.mtg_search_agent.models.search import SearchQuery
        default_search_query = SearchQuery(query="c:r", explanation="Mocked search", confidence=0.8)

        mock_query_result = Mock()
        mock_query_result.output = default_search_query
        mock_query_agent.run = AsyncMock(return_value=mock_query_result)
        mock_query_agent.run_sync = Mock(return_value=mock_query_result)
        mock_query_agent.run_stream = AsyncMock(return_value=mock_query_result)

        # Mock the evaluation agent with realistic return types
        from src.mtg_search_agent.models.evaluation import LightweightAgentResult, LightweightCardScore
        default_lightweight_result = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(card_id="mock-id", name="Mock Card", score=7, reasoning="Mocked evaluation")
            ],
            feedback_for_query_agent="Mocked feedback"
        )

        mock_eval_result = Mock()
        mock_eval_result.output = default_lightweight_result
        mock_eval_agent.run = AsyncMock(return_value=mock_eval_result)
        mock_eval_agent.run_sync = Mock(return_value=mock_eval_result)
        mock_eval_agent.run_stream = AsyncMock(return_value=mock_eval_result)

        # Mock the feedback synthesis agent
        mock_feedback_result = Mock()
        mock_feedback_result.output = "Mocked feedback"
        mock_feedback_agent.run = AsyncMock(return_value=mock_feedback_result)
        mock_feedback_agent.run_sync = Mock(return_value=mock_feedback_result)

        yield {
            'query_agent': mock_query_agent,
            'eval_agent': mock_eval_agent,
            'feedback_agent': mock_feedback_agent
        }

# Import models for fixtures
from src.mtg_search_agent.models.card import Card
from src.mtg_search_agent.models.search import SearchQuery, SearchResult, TagSuggestion
from src.mtg_search_agent.models.evaluation import (
    CardScore, LightweightCardScore, EvaluationResult, LightweightEvaluationResult
)
from src.mtg_search_agent.events import SearchEventEmitter


# Pytest configuration
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== CARD FIXTURES ====================

@pytest.fixture
def sample_card_data():
    """Sample MTG card data for testing - matches Scryfall API format."""
    return {
        "id": "b1b77b57-5b4c-4b35-8c9a-9f5e5e5e5e5e",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "cmc": 1.0,
        "type_line": "Instant",
        "oracle_text": "Lightning Bolt deals 3 damage to any target.",
        "colors": ["R"],
        "color_identity": ["R"],
        "keywords": [],
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal"
        },
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "rarity": "common",
        "collector_number": "161",
        "artist": "Christopher Rush",
        "scryfall_uri": "https://scryfall.com/card/lea/161/lightning-bolt",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/front/b/1/b1b77b57.jpg",
            "normal": "https://cards.scryfall.io/normal/front/b/1/b1b77b57.jpg",
            "large": "https://cards.scryfall.io/large/front/b/1/b1b77b57.jpg"
        },
        "prices": {
            "usd": "0.50",
            "usd_foil": "2.00",
            "eur": "0.45",
            "tix": "0.02"
        }
    }


@pytest.fixture
def sample_creature_data():
    """Sample creature card data for testing."""
    return {
        "id": "c2c22c22-2c2c-2c2c-2c2c-2c2c2c2c2c2c",
        "name": "Serra Angel",
        "mana_cost": "{3}{W}{W}",
        "cmc": 5.0,
        "type_line": "Creature — Angel",
        "oracle_text": "Flying, vigilance",
        "power": "4",
        "toughness": "4",
        "colors": ["W"],
        "color_identity": ["W"],
        "keywords": ["Flying", "Vigilance"],
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal"
        },
        "set": "lea",
        "set_name": "Limited Edition Alpha",
        "rarity": "uncommon",
        "collector_number": "38",
        "artist": "Douglas Shuler",
        "scryfall_uri": "https://scryfall.com/card/lea/38/serra-angel",
        "image_uris": {
            "normal": "https://cards.scryfall.io/normal/front/c/2/c2c22c22.jpg"
        },
        "prices": {
            "usd": "1.50",
            "usd_foil": None,
            "eur": "1.25",
            "tix": "0.05"
        }
    }


@pytest.fixture
def sample_card(sample_card_data):
    """Sample Card model instance."""
    return Card.from_scryfall(sample_card_data)


@pytest.fixture
def sample_creature(sample_creature_data):
    """Sample creature Card model instance."""
    return Card.from_scryfall(sample_creature_data)


@pytest.fixture
def multiple_cards(sample_card_data, sample_creature_data):
    """List of multiple Card instances for testing."""
    cards_data = [sample_card_data, sample_creature_data]
    # Add some variations
    card3_data = sample_card_data.copy()
    card3_data.update({
        "id": "d3d33d33-3d3d-3d3d-3d3d-3d3d3d3d3d3d",
        "name": "Shock",
        "oracle_text": "Shock deals 2 damage to any target."
    })
    cards_data.append(card3_data)

    return [Card.from_scryfall(data) for data in cards_data]


# ==================== SEARCH FIXTURES ====================

@pytest.fixture
def sample_search_query():
    """Sample search query for testing."""
    return "red instant damage"


@pytest.fixture
def sample_search_query_model():
    """Sample SearchQuery model instance."""
    return SearchQuery(
        query="c:r t:instant",
        explanation="Search for red instant spells",
        confidence=0.8
    )


@pytest.fixture
def sample_search_result(sample_search_query_model, multiple_cards):
    """Sample SearchResult model instance."""
    return SearchResult(
        query=sample_search_query_model,
        cards=multiple_cards,
        total_cards=len(multiple_cards),
        has_more=False
    )


@pytest.fixture
def sample_tag_suggestions():
    """Sample tag suggestions for testing."""
    return [
        TagSuggestion(tag="instant", score=0.95, category="type"),
        TagSuggestion(tag="damage", score=0.85, category="mechanic"),
        TagSuggestion(tag="red", score=0.90, category="color")
    ]


# ==================== EVALUATION FIXTURES ====================

@pytest.fixture
def sample_card_score(sample_card):
    """Sample CardScore instance."""
    return CardScore(
        card=sample_card,
        score=8,
        reasoning="Great direct damage spell, very relevant to the search query."
    )


@pytest.fixture
def sample_lightweight_card_score():
    """Sample LightweightCardScore instance."""
    return LightweightCardScore(
        card_id="b1b77b57-5b4c-4b35-8c9a-9f5e5e5e5e5e",
        name="Lightning Bolt",
        score=8,
        reasoning="Great direct damage spell, very relevant to the search query."
    )


@pytest.fixture
def sample_evaluation_result(sample_card_score, sample_creature):
    """Sample EvaluationResult instance."""
    # Create additional scored cards
    creature_score = CardScore(
        card=sample_creature,
        score=6,
        reasoning="Solid creature but not directly related to damage spells."
    )

    scored_cards = [sample_card_score, creature_score]
    average_score = sum(sc.score for sc in scored_cards) / len(scored_cards)

    return EvaluationResult(
        scored_cards=scored_cards,
        average_score=average_score,
        should_continue=False,
        feedback_for_query_agent="Results are satisfactory for red damage spells.",
        iteration_count=1
    )


@pytest.fixture
def sample_lightweight_evaluation_result(sample_lightweight_card_score):
    """Sample LightweightEvaluationResult instance."""
    lightweight_scores = [
        sample_lightweight_card_score,
        LightweightCardScore(
            card_id="c2c22c22-2c2c-2c2c-2c2c-2c2c2c2c2c2c",
            name="Serra Angel",
            score=6,
            reasoning="Solid creature but not directly related to damage spells."
        )
    ]

    average_score = sum(sc.score for sc in lightweight_scores) / len(lightweight_scores)

    return LightweightEvaluationResult(
        scored_cards=lightweight_scores,
        average_score=average_score,
        should_continue=False,
        feedback_for_query_agent="Results are satisfactory for red damage spells.",
        iteration_count=1
    )


# ==================== EVENT FIXTURES ====================

@pytest.fixture
def mock_event_emitter():
    """Mock event emitter for testing."""
    emitter = Mock(spec=SearchEventEmitter)
    emitter.emit = Mock()
    emitter.emit_async = AsyncMock()
    emitter.on = Mock()
    emitter.off = Mock()
    emitter.clear_listeners = Mock()
    return emitter


@pytest.fixture
def real_event_emitter():
    """Real event emitter instance for testing."""
    return SearchEventEmitter()


# ==================== API MOCK FIXTURES ====================

@pytest.fixture
def mock_scryfall_response(sample_card_data, sample_creature_data):
    """Mock successful Scryfall API response."""
    return {
        "object": "list",
        "total_cards": 2,
        "has_more": False,
        "data": [sample_card_data, sample_creature_data]
    }


@pytest.fixture
def mock_scryfall_paginated_response(sample_card_data):
    """Mock paginated Scryfall API response."""
    return {
        "object": "list",
        "total_cards": 175,
        "has_more": True,
        "next_page": "https://api.scryfall.com/cards/search?q=c%3Ar&page=2",
        "data": [sample_card_data]
    }


@pytest.fixture
def mock_requests_session():
    """Mock requests session for API testing."""
    session = Mock()
    session.headers = {}
    session.get = Mock()
    return session


# ==================== AGENT MOCK FIXTURES ====================

@pytest.fixture
def mock_openai_model():
    """Mock OpenAI model for agent testing."""
    model = Mock()
    model.agent_model = "gpt-5-mini"
    return model


@pytest.fixture
def mock_query_agent_response(sample_search_query_model):
    """Mock response from query agent."""
    return sample_search_query_model


@pytest.fixture
def mock_evaluation_agent_response(sample_lightweight_evaluation_result):
    """Mock response from evaluation agent."""
    return sample_lightweight_evaluation_result


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-api-key-12345",
        "SCRYFALL_RATE_LIMIT_MS": "50",
        "MAX_SEARCH_LOOPS": "3",
        "ENABLE_PARALLEL_EVALUATION": "true"
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def reset_config_module():
    """Reset config module state between tests."""
    # Store original values
    import src.mtg_search_agent.config as config
    original_values = {}
    for attr in dir(config):
        if not attr.startswith('_') and attr.isupper():
            original_values[attr] = getattr(config, attr)

    yield

    # Restore original values
    for attr, value in original_values.items():
        setattr(config, attr, value)


# ==================== TAG SEARCH FIXTURES ====================

@pytest.fixture
def sample_tags_data():
    """Sample tags data for testing tag search functionality."""
    return {
        "types": ["instant", "sorcery", "creature", "artifact", "enchantment"],
        "colors": ["white", "blue", "black", "red", "green"],
        "mechanics": ["flying", "trample", "haste", "vigilance", "damage"],
        "keywords": ["flash", "lifelink", "deathtouch", "first strike"]
    }


@pytest.fixture
def mock_tags_file(sample_tags_data, tmp_path):
    """Create a temporary tags.json file for testing."""
    import json
    tags_file = tmp_path / "tags.json"
    tags_file.write_text(json.dumps(sample_tags_data, indent=2))
    return str(tags_file)


# ==================== ASYNC TESTING HELPERS ====================

@pytest.fixture
def anyio_backend():
    """Configure anyio backend for async testing."""
    return "asyncio"


# ==================== TIME/DURATION FIXTURES ====================

@pytest.fixture
def fixed_time():
    """Fixed time for consistent testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def mock_time():
    """Mock time.time() for duration testing."""
    with patch('time.time') as mock_time_func:
        mock_time_func.return_value = 1704110400.0  # 2024-01-01 12:00:00 UTC
        yield mock_time_func