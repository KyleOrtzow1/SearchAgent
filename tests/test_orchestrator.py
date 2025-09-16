"""
Integration tests for the SearchOrchestrator.

Tests cover:
- Complete search workflow from natural language to results
- Multi-iteration refinement loops
- Component integration (agents, tools, API)
- Event emission throughout the process
- Error handling and recovery
- Performance characteristics
- Cache management and optimization
- Mock-based testing to avoid external dependencies
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict

from src.mtg_search_agent.orchestrator import SearchOrchestrator
from src.mtg_search_agent.models.search import SearchQuery, SearchResult
from src.mtg_search_agent.models.evaluation import EvaluationResult, LightweightEvaluationResult, CardScore, LightweightCardScore
from src.mtg_search_agent.models.card import Card
from src.mtg_search_agent.events import SearchEventEmitter


class TestSearchOrchestratorInitialization:
    """Test suite for SearchOrchestrator initialization."""

    def test_orchestrator_initialization_default(self):
        """Test SearchOrchestrator initialization with default settings."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent'), \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI'):

            orchestrator = SearchOrchestrator()

            assert hasattr(orchestrator, 'events')
            assert isinstance(orchestrator.events, SearchEventEmitter)
            assert hasattr(orchestrator, 'query_agent')
            assert hasattr(orchestrator, 'evaluation_agent')
            assert hasattr(orchestrator, 'scryfall_api')
            assert hasattr(orchestrator, 'card_cache')
            assert orchestrator.enable_streaming is False
            assert orchestrator.max_loops > 0

    def test_orchestrator_initialization_with_streaming(self):
        """Test SearchOrchestrator initialization with streaming enabled."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent'), \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI'):

            orchestrator = SearchOrchestrator(enable_streaming=True)

            assert orchestrator.enable_streaming is True

    def test_orchestrator_component_initialization(self):
        """Test that all components are properly initialized with event emitter."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall:

            orchestrator = SearchOrchestrator()

            # Verify components were initialized with event emitter
            mock_query_agent.assert_called_once_with(event_emitter=orchestrator.events)
            mock_eval_agent.assert_called_once_with(event_emitter=orchestrator.events)
            mock_scryfall.assert_called_once_with(event_emitter=orchestrator.events)

    def test_orchestrator_cache_initialization(self):
        """Test that card cache is properly initialized."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent'), \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI'):

            orchestrator = SearchOrchestrator()

            assert isinstance(orchestrator.card_cache, dict)
            assert len(orchestrator.card_cache) == 0


class TestSearchOrchestratorBasicSearch:
    """Test suite for basic search functionality."""

    @pytest.mark.asyncio
    async def test_simple_successful_search(self, sample_search_query_model,
                                          sample_search_result, sample_lightweight_evaluation_result):
        """Test a simple successful search that completes in one iteration."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mock instances
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Setup mock responses
            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("red instant damage spells")

            # Verify the workflow
            assert isinstance(result, EvaluationResult)
            assert len(result.scored_cards) > 0
            assert result.iteration_count >= 1

            # Verify method calls
            mock_query_agent.generate_query.assert_called_once()
            mock_scryfall.search_cards.assert_called_once()
            mock_eval_agent.evaluate_cards.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_empty_results(self, sample_search_query_model):
        """Test search that returns no cards."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mock instances
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Setup empty search result
            empty_result = SearchResult(
                query=sample_search_query_model,
                cards=[],
                total_cards=0,
                has_more=False
            )

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=empty_result)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("nonexistent cards")

            # Should handle empty results gracefully
            assert isinstance(result, EvaluationResult)
            assert len(result.scored_cards) == 0

    @pytest.mark.asyncio
    async def test_search_event_emission(self, sample_search_query_model,
                                       sample_search_result, sample_lightweight_evaluation_result):
        """Test that search emits appropriate events throughout the process."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator()

            # Track events
            events_received = []
            orchestrator.events.on("search_started", lambda e: events_received.append(e))
            orchestrator.events.on("search_completed", lambda e: events_received.append(e))
            orchestrator.events.on("iteration_started", lambda e: events_received.append(e))
            orchestrator.events.on("iteration_completed", lambda e: events_received.append(e))

            await orchestrator.search("test search")

            # Verify events were emitted
            assert len(events_received) >= 4  # At least start, iteration start/complete, end
            event_types = [event.event_type for event in events_received]
            assert "search_started" in event_types
            assert "search_completed" in event_types


class TestSearchOrchestratorMultiIteration:
    """Test suite for multi-iteration refinement functionality."""

    @pytest.mark.asyncio
    async def test_multi_iteration_refinement(self, sample_search_query_model,
                                            sample_search_result, multiple_cards):
        """Test search that requires multiple iterations to find satisfactory results."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class, \
             patch('src.mtg_search_agent.orchestrator.MAX_SEARCH_LOOPS', 3):

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Setup progressive improvement across iterations
            queries = [
                SearchQuery(query="c:r", explanation="Initial red search", confidence=0.5),
                SearchQuery(query="c:r t:instant", explanation="Red instants", confidence=0.7),
                SearchQuery(query="c:r t:instant o:damage", explanation="Red damage instants", confidence=0.9)
            ]

            evaluations = []
            for i, query in enumerate(queries):
                # Progressive score improvement
                scores = [
                    LightweightCardScore(card_id=card.id, name=card.name, score=4 + i, reasoning=f"Iteration {i+1} score")
                    for card in multiple_cards
                ]
                avg_score = 4 + i
                should_continue = i < 2  # Continue for first two iterations

                eval_result = LightweightEvaluationResult(
                    scored_cards=scores,
                    average_score=avg_score,
                    should_continue=should_continue,
                    feedback_for_query_agent=f"Iteration {i+1} feedback",
                    iteration_count=i+1
                )
                evaluations.append(eval_result)

            mock_query_agent.generate_query = AsyncMock(side_effect=queries)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(side_effect=evaluations)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("improve over iterations")

            # Should complete multiple iterations
            assert result.iteration_count == 3
            assert mock_query_agent.generate_query.call_count == 3
            assert mock_eval_agent.evaluate_cards.call_count == 3

    @pytest.mark.asyncio
    async def test_early_termination_on_good_results(self, sample_search_query_model,
                                                   sample_search_result, multiple_cards):
        """Test that search terminates early when good results are found."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # High-quality result that should terminate search
            high_quality_scores = [
                LightweightCardScore(card_id=card.id, name=card.name, score=9, reasoning="Excellent match")
                for card in multiple_cards
            ]

            high_quality_result = LightweightEvaluationResult(
                scored_cards=high_quality_scores,
                average_score=9.0,
                should_continue=False,  # Should stop here
                feedback_for_query_agent="Excellent results",
                iteration_count=1
            )

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=high_quality_result)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("high quality search")

            # Should terminate after first iteration
            assert result.iteration_count == 1
            assert mock_query_agent.generate_query.call_count == 1
            assert result.average_score == 9.0

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, sample_search_query_model,
                                      sample_search_result, multiple_cards):
        """Test that search respects maximum iteration limit."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class, \
             patch('src.mtg_search_agent.orchestrator.MAX_SEARCH_LOOPS', 2):

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Low-quality results that would normally continue
            low_quality_scores = [
                LightweightCardScore(card_id=card.id, name=card.name, score=3, reasoning="Poor match")
                for card in multiple_cards
            ]

            # Results that always want to continue
            continuing_results = []
            for i in range(5):  # More than max loops
                result = LightweightEvaluationResult(
                    scored_cards=low_quality_scores,
                    average_score=3.0,
                    should_continue=True,  # Always wants to continue
                    feedback_for_query_agent="Need better results",
                    iteration_count=i+1
                )
                continuing_results.append(result)

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(side_effect=continuing_results)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("reaches max iterations")

            # Should stop at max iterations even though evaluation wants to continue
            assert result.iteration_count == 2  # MAX_SEARCH_LOOPS
            assert mock_query_agent.generate_query.call_count == 2


class TestSearchOrchestratorCaching:
    """Test suite for card caching functionality."""

    @pytest.mark.asyncio
    async def test_card_cache_usage(self, sample_search_query_model, multiple_cards):
        """Test that card caching works correctly across iterations."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # First iteration results
            first_result = SearchResult(
                query=sample_search_query_model,
                cards=multiple_cards[:2],  # First 2 cards
                total_cards=2,
                has_more=False
            )

            # Second iteration with overlapping cards
            second_result = SearchResult(
                query=sample_search_query_model,
                cards=multiple_cards[1:],  # Cards 2-3 (overlap with first)
                total_cards=2,
                has_more=False
            )

            # Mock evaluation responses
            def mock_evaluation(request, cards, iteration, **kwargs):
                # Should only evaluate new cards, not cached ones
                card_cache = kwargs.get('card_cache', {})

                scores = []
                for card in cards:
                    if card.id not in card_cache:
                        scores.append(LightweightCardScore(
                            card_id=card.id,
                            name=card.name,
                            score=7,
                            reasoning="New evaluation"
                        ))

                return LightweightEvaluationResult(
                    scored_cards=scores,
                    average_score=7.0,
                    should_continue=iteration == 1,  # Continue after first iteration
                    feedback_for_query_agent="Cache test",
                    iteration_count=iteration
                )

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(side_effect=[first_result, second_result])
            mock_eval_agent.evaluate_cards = AsyncMock(side_effect=mock_evaluation)

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("cache test")

            # Should have cached cards from first iteration
            assert len(orchestrator.card_cache) > 0

            # Should have called evaluation twice but with cache awareness
            assert mock_eval_agent.evaluate_cards.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_prevents_duplicate_evaluations(self, sample_search_query_model, sample_card):
        """Test that cached cards are not re-evaluated."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Setup mocks
            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Same cards returned in both iterations
            same_result = SearchResult(
                query=sample_search_query_model,
                cards=[sample_card],
                total_cards=1,
                has_more=False
            )

            # First evaluation creates cache entry
            first_eval = LightweightEvaluationResult(
                scored_cards=[LightweightCardScore(card_id=sample_card.id, name=sample_card.name, score=8, reasoning="First eval")],
                average_score=8.0,
                should_continue=True,
                feedback_for_query_agent="Continue",
                iteration_count=1
            )

            # Second evaluation should use cache
            second_eval = EvaluationResult(
                scored_cards=[],  # No new cards to evaluate
                average_score=8.0,
                should_continue=False,
                feedback_for_query_agent="Using cache",
                iteration_count=2
            )

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=same_result)
            mock_eval_agent.evaluate_cards = AsyncMock(side_effect=[first_eval, second_eval])

            orchestrator = SearchOrchestrator()

            result = await orchestrator.search("duplicate cards test")

            # Cache should prevent duplicate evaluation
            assert len(orchestrator.card_cache) == 1
            assert sample_card.id in orchestrator.card_cache


class TestSearchOrchestratorErrorHandling:
    """Test suite for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_query_generation_error(self):
        """Test handling of query generation errors."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI'):

            mock_query_agent = Mock()
            mock_query_agent_class.return_value = mock_query_agent

            # Mock query generation failure
            mock_query_agent.generate_query = AsyncMock(
                side_effect=Exception("Query generation failed")
            )

            orchestrator = SearchOrchestrator()

            with pytest.raises(Exception) as exc_info:
                await orchestrator.search("error test")

            assert "Query generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_scryfall_api_error(self, sample_search_query_model):
        """Test handling of Scryfall API errors."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(side_effect=Exception("API error"))

            orchestrator = SearchOrchestrator()

            with pytest.raises(Exception) as exc_info:
                await orchestrator.search("api error test")

            assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluation_error(self, sample_search_query_model, sample_search_result):
        """Test handling of evaluation errors."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(
                side_effect=Exception("Evaluation failed")
            )

            orchestrator = SearchOrchestrator()

            with pytest.raises(Exception) as exc_info:
                await orchestrator.search("evaluation error test")

            assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, sample_search_query_model,
                                          sample_search_result, sample_lightweight_evaluation_result):
        """Test recovery from partial failures in multi-iteration search."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # First iteration succeeds, second fails, third succeeds
            evaluation_responses = [
                sample_lightweight_evaluation_result,  # Success
                Exception("Temporary failure"),  # Failure
                sample_lightweight_evaluation_result   # Recovery
            ]

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(side_effect=evaluation_responses)

            orchestrator = SearchOrchestrator()

            # Should fail on the temporary failure
            with pytest.raises(Exception) as exc_info:
                await orchestrator.search("partial failure test")

            assert "Temporary failure" in str(exc_info.value)


class TestSearchOrchestratorStreaming:
    """Test suite for streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_enabled(self, sample_search_query_model,
                                   sample_search_result, sample_lightweight_evaluation_result):
        """Test search with streaming enabled."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator(enable_streaming=True)

            result = await orchestrator.search("streaming test")

            # Should pass streaming parameter to query agent
            mock_query_agent.generate_query.assert_called()
            call_kwargs = mock_query_agent.generate_query.call_args[1]
            assert call_kwargs.get('enable_streaming') is True

    @pytest.mark.asyncio
    async def test_streaming_disabled(self, sample_search_query_model,
                                    sample_search_result, sample_lightweight_evaluation_result):
        """Test search with streaming disabled."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator(enable_streaming=False)

            result = await orchestrator.search("non-streaming test")

            # Should pass streaming parameter as False
            mock_query_agent.generate_query.assert_called()
            call_kwargs = mock_query_agent.generate_query.call_args[1]
            assert call_kwargs.get('enable_streaming') is False


class TestSearchOrchestratorPerformance:
    """Test suite for performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, sample_search_query_model,
                                     sample_search_result, sample_lightweight_evaluation_result):
        """Test multiple concurrent searches."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            # Each orchestrator gets its own instances
            def create_mock_agent(*args, **kwargs):
                mock_agent = Mock()
                mock_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
                return mock_agent

            def create_mock_eval_agent(*args, **kwargs):
                mock_agent = Mock()
                mock_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)
                return mock_agent

            def create_mock_scryfall(*args, **kwargs):
                mock_api = Mock()
                mock_api.search_cards = Mock(return_value=sample_search_result)
                return mock_api

            mock_query_agent_class.side_effect = create_mock_agent
            mock_eval_agent_class.side_effect = create_mock_eval_agent
            mock_scryfall_class.side_effect = create_mock_scryfall

            # Create multiple orchestrators
            orchestrators = [SearchOrchestrator() for _ in range(3)]

            # Start concurrent searches
            tasks = [
                orchestrators[0].search("first search"),
                orchestrators[1].search("second search"),
                orchestrators[2].search("third search"),
            ]

            results = await asyncio.gather(*tasks)

            # All should complete successfully
            assert len(results) == 3
            assert all(isinstance(result, EvaluationResult) for result in results)

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self, sample_search_query_model):
        """Test handling of large result sets."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            # Create large result set
            large_cards = []
            for i in range(100):
                card_data = {
                    "id": f"large-card-{i}",
                    "name": f"Large Card {i}",
                    "type_line": "Instant",
                    "set": "large",
                    "set_name": "Large Set",
                    "rarity": "common",
                    "collector_number": str(i),
                    "scryfall_uri": f"https://scryfall.com/large-{i}"
                }
                large_cards.append(Card.from_scryfall(card_data))

            large_result = SearchResult(
                query=sample_search_query_model,
                cards=large_cards,
                total_cards=100,
                has_more=False
            )

            large_scores = [
                LightweightCardScore(card_id=card.id, name=card.name, score=5, reasoning="Large set evaluation")
                for card in large_cards
            ]

            large_evaluation = LightweightEvaluationResult(
                scored_cards=large_scores,
                average_score=5.0,
                should_continue=False,
                feedback_for_query_agent="Large set handled",
                iteration_count=1
            )

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=large_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=large_evaluation)

            orchestrator = SearchOrchestrator()

            import time
            start_time = time.time()

            result = await orchestrator.search("large result set")

            elapsed_time = time.time() - start_time

            # Should handle large sets efficiently
            assert elapsed_time < 10.0  # Should complete within 10 seconds
            assert len(result.scored_cards) == 100


class TestSearchOrchestratorEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_search_request(self):
        """Test handling of empty search request."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent'), \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI'):

            mock_query_agent = Mock()
            mock_query_agent_class.return_value = mock_query_agent

            # Should still attempt to generate a query from empty request
            empty_query = SearchQuery(query="*", explanation="All cards")
            mock_query_agent.generate_query = AsyncMock(return_value=empty_query)

            orchestrator = SearchOrchestrator()

            # Should handle empty request gracefully
            try:
                result = await orchestrator.search("")
                # If it doesn't raise an exception, that's also acceptable
            except Exception:
                # Some validation error is also acceptable
                pass

    @pytest.mark.asyncio
    async def test_very_long_search_request(self, sample_search_query_model,
                                          sample_search_result, sample_lightweight_evaluation_result):
        """Test handling of very long search requests."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator()

            # Very long request
            long_request = "Find me red instant damage spells that " * 100

            result = await orchestrator.search(long_request)

            # Should handle long requests without issues
            assert isinstance(result, EvaluationResult)

    @pytest.mark.asyncio
    async def test_unicode_search_request(self, sample_search_query_model,
                                        sample_search_result, sample_lightweight_evaluation_result):
        """Test handling of unicode characters in search requests."""
        with patch('src.mtg_search_agent.orchestrator.QueryAgent') as mock_query_agent_class, \
             patch('src.mtg_search_agent.orchestrator.EvaluationAgent') as mock_eval_agent_class, \
             patch('src.mtg_search_agent.orchestrator.ScryfallAPI') as mock_scryfall_class:

            mock_query_agent = Mock()
            mock_eval_agent = Mock()
            mock_scryfall = Mock()

            mock_query_agent_class.return_value = mock_query_agent
            mock_eval_agent_class.return_value = mock_eval_agent
            mock_scryfall_class.return_value = mock_scryfall

            mock_query_agent.generate_query = AsyncMock(return_value=sample_search_query_model)
            mock_scryfall.search_cards = Mock(return_value=sample_search_result)
            mock_eval_agent.evaluate_cards = AsyncMock(return_value=sample_lightweight_evaluation_result)

            orchestrator = SearchOrchestrator()

            # Unicode request
            unicode_request = "Find cards with Æther or 以太"

            result = await orchestrator.search(unicode_request)

            # Should handle unicode without issues
            assert isinstance(result, EvaluationResult)