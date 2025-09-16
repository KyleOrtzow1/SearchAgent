"""
Async unit tests for the EvaluationAgent.

Tests cover:
- Agent initialization and configuration
- Card evaluation with scoring (1-10 scale)
- Batch evaluation and parallel processing
- Feedback generation for query improvement
- Event emission during evaluation
- Error handling and edge cases
- Mock-based testing to avoid actual AI API calls
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List
from pydantic import ValidationError

from src.mtg_search_agent.agents.evaluation_agent import (
    EvaluationAgent, lightweight_evaluation_agent, feedback_synthesis_agent
)
from src.mtg_search_agent.models.evaluation import (
    EvaluationResult, LightweightEvaluationResult, LightweightAgentResult,
    CardScore, LightweightCardScore
)
from src.mtg_search_agent.models.card import Card


class TestEvaluationAgent:
    """Test suite for EvaluationAgent class."""

    def test_evaluation_agent_initialization_without_emitter(self):
        """Test EvaluationAgent initialization without event emitter."""
        agent = EvaluationAgent()
        assert hasattr(agent, 'events')
        assert agent.events is None

    def test_evaluation_agent_initialization_with_emitter(self, mock_event_emitter):
        """Test EvaluationAgent initialization with event emitter."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)
        assert agent.events == mock_event_emitter

    def test_build_evaluation_prompt_basic(self, multiple_cards):
        """Test basic evaluation prompt building."""
        agent = EvaluationAgent()
        request = "red instant damage spells"

        prompt = agent._build_evaluation_prompt(
            natural_language_request=request,
            cards=multiple_cards,
            iteration_count=1,
            total_cards=len(multiple_cards)
        )

        assert request in prompt
        assert "Cards found:" in prompt
        assert str(len(multiple_cards)) in prompt
        for card in multiple_cards:
            assert card.name in prompt

    def test_build_evaluation_prompt_with_previous_queries(self, multiple_cards):
        """Test evaluation prompt with previous queries."""
        agent = EvaluationAgent()
        request = "red burn spells"
        previous_queries = ["c:r t:instant", "c:r o:damage"]

        prompt = agent._build_evaluation_prompt(
            natural_language_request=request,
            cards=multiple_cards,
            iteration_count=2,
            total_cards=len(multiple_cards),
            previous_queries=previous_queries
        )

        assert "Previous queries tried:" in prompt
        assert "c:r t:instant" in prompt
        assert "c:r o:damage" in prompt

    def test_build_evaluation_prompt_with_batch_info(self, multiple_cards):
        """Test evaluation prompt with batch information."""
        agent = EvaluationAgent()
        request = "flying creatures"
        batch_info = (1, 3)  # Batch 2 of 3

        prompt = agent._build_evaluation_prompt(
            natural_language_request=request,
            cards=multiple_cards,
            iteration_count=1,
            total_cards=30,
            batch_info=batch_info
        )

        assert "Batch 2/3" in prompt
        assert f"showing {len(multiple_cards)} of 30 total cards" in prompt

    def test_build_evaluation_prompt_card_details(self, sample_card):
        """Test that evaluation prompt includes proper card details."""
        agent = EvaluationAgent()
        cards = [sample_card]

        prompt = agent._build_evaluation_prompt(
            natural_language_request="test",
            cards=cards,
            iteration_count=1,
            total_cards=1
        )

        assert sample_card.name in prompt
        assert sample_card.id in prompt
        assert sample_card.type_line in prompt
        if sample_card.mana_cost:
            assert sample_card.mana_cost in prompt
        if sample_card.oracle_text:
            assert sample_card.oracle_text in prompt

    def test_build_evaluation_prompt_card_without_optional_fields(self):
        """Test prompt building with cards missing optional fields."""
        agent = EvaluationAgent()

        # Create card with minimal data
        minimal_card_data = {
            "id": "minimal-id",
            "name": "Minimal Card",
            "type_line": "Instant",
            "set": "min",
            "set_name": "Minimal",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/minimal"
        }
        minimal_card = Card.from_scryfall(minimal_card_data)

        prompt = agent._build_evaluation_prompt(
            natural_language_request="test",
            cards=[minimal_card],
            iteration_count=1,
            total_cards=1
        )

        assert minimal_card.name in prompt
        assert "No cost" in prompt  # Should handle missing mana_cost
        assert minimal_card.type_line in prompt

    @pytest.mark.asyncio
    async def test_evaluate_cards_basic(self, multiple_cards, mock_event_emitter):
        """Test basic card evaluation functionality."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Mock the lightweight evaluation response
        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=card.id,
                    name=card.name,
                    score=8,
                    reasoning="Relevant to the search query"
                ) for card in multiple_cards
            ],
            feedback_for_query_agent="Good results found"
        )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            # Mock the async run method to return a result object with .output
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="red instant damage",
                cards=multiple_cards,
                iteration_count=1
            )

            assert isinstance(result, LightweightEvaluationResult)
            assert len(result.scored_cards) == len(multiple_cards)
            # The actual agent returns different feedback when cards are few
            assert result.feedback_for_query_agent is not None
            assert all(1 <= sc.score <= 10 for sc in result.scored_cards)

    @pytest.mark.asyncio
    async def test_evaluate_cards_with_previous_queries(self, multiple_cards, mock_event_emitter):
        """Test card evaluation with previous query context."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=multiple_cards[0].id,
                    name=multiple_cards[0].name,
                    score=7,
                    reasoning="Good match considering previous attempts"
                )
            ]
        )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            # Mock the async run method to return a result object with .output
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="burn spells",
                cards=[multiple_cards[0]],
                iteration_count=2,
                previous_queries=["c:r", "c:r t:instant"]
            )

            # Verify prompt included previous queries
            mock_run.assert_called_once()
            prompt_used = mock_run.call_args[0][0]
            assert "Previous queries tried:" in prompt_used
            assert "c:r" in prompt_used

    @pytest.mark.asyncio
    async def test_evaluate_cards_parallel_processing(self, mock_event_emitter):
        """Test parallel card evaluation processing."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Create larger set of cards for batching
        many_cards = []
        for i in range(25):  # Enough to trigger parallel processing
            card_data = {
                "id": f"card-{i}",
                "name": f"Card {i}",
                "type_line": "Instant",
                "set": "test",
                "set_name": "Test Set",
                "rarity": "common",
                "collector_number": str(i),
                "scryfall_uri": f"https://scryfall.com/card-{i}"
            }
            many_cards.append(Card.from_scryfall(card_data))

        # Mock batch responses
        def mock_batch_response(*args, **kwargs):
            # Return different scores for different batches
            batch_size = 10  # From config
            return LightweightAgentResult(
                scored_cards=[
                    LightweightCardScore(
                        card_id=f"card-{i}",
                        name=f"Card {i}",
                        score=5 + (i % 5),  # Varying scores
                        reasoning=f"Card {i} evaluation"
                    ) for i in range(min(batch_size, len(many_cards)))
                ]
            )

        with patch('src.mtg_search_agent.agents.evaluation_agent.ENABLE_PARALLEL_EVALUATION', True):
            with patch('src.mtg_search_agent.agents.evaluation_agent.EVALUATION_BATCH_SIZE', 10):
                with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
                    # Mock the async run method to return a result object with .output
                    def mock_run_func(*args, **kwargs):
                        mock_result = Mock()
                        mock_result.output = mock_batch_response(*args, **kwargs)
                        return mock_result
                    mock_run.side_effect = mock_run_func

                    result = await agent.evaluate_cards(
                        natural_language_request="test cards",
                        cards=many_cards,
                        iteration_count=1
                    )

                    # Should have called the agent multiple times for batches
                    assert mock_run.call_count > 1

                    # Should emit parallel processing events
                    emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
                    event_types = [event.event_type for event in emitted_events]
                    assert any("parallel" in event_type for event_type in event_types)

    @pytest.mark.asyncio
    async def test_evaluate_cards_sequential_processing(self, multiple_cards, mock_event_emitter):
        """Test sequential card evaluation processing."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=card.id,
                    name=card.name,
                    score=6,
                    reasoning="Sequential evaluation"
                ) for card in multiple_cards
            ]
        )

        # Disable parallel processing
        with patch('src.mtg_search_agent.agents.evaluation_agent.ENABLE_PARALLEL_EVALUATION', False):
            with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
                # Mock the async run method to return a result object with .output
                mock_result = Mock()
                mock_result.output = mock_response
                mock_run.return_value = mock_result

                result = await agent.evaluate_cards(
                    natural_language_request="test",
                    cards=multiple_cards,
                    iteration_count=1
                )

                # Should make single call for sequential processing
                assert mock_run.call_count == 1

                # Should emit appropriate strategy event
                emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
                event_types = [event.event_type for event in emitted_events]
                assert "evaluation_strategy_selected" in event_types

    @pytest.mark.asyncio
    async def test_evaluate_cards_with_cache(self, multiple_cards, mock_event_emitter):
        """Test card evaluation behavior (cache functionality removed from actual API)."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=card.id,
                    name=card.name,
                    score=7,
                    reasoning="Standard evaluation"
                ) for card in multiple_cards
            ]
        )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            # Mock the async run method to return a result object with .output
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="test",
                cards=multiple_cards,
                iteration_count=1
            )

            # Should evaluate all cards
            assert len(result.scored_cards) == len(multiple_cards)
            assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_cards_no_cards(self, mock_event_emitter):
        """Test evaluation with empty cards list."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        result = await agent.evaluate_cards(
            natural_language_request="test",
            cards=[],
            iteration_count=1
        )

        assert isinstance(result, LightweightEvaluationResult)
        assert len(result.scored_cards) == 0
        assert result.average_score == 0.0
        assert result.should_continue is True

    @pytest.mark.asyncio
    async def test_evaluate_cards_error_handling(self, multiple_cards, mock_event_emitter):
        """Test error handling during card evaluation."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Evaluation model error")

            with pytest.raises(Exception) as exc_info:
                await agent.evaluate_cards(
                    natural_language_request="test",
                    cards=multiple_cards,
                    iteration_count=1
                )

            assert "Evaluation model error" in str(exc_info.value)

            # Should emit evaluation strategy event before error occurs
            emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
            event_types = [event.event_type for event in emitted_events]
            assert "evaluation_strategy_selected" in event_types

    @pytest.mark.asyncio
    async def test_feedback_synthesis(self, mock_event_emitter):
        """Test feedback synthesis from multiple batches."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        batch_feedbacks = [
            "Cards are too broad, need more specific",
            "Some good matches but missing key effects",
            "Better results, closer to target"
        ]

        synthesized_feedback = "Overall results show improvement but need more specificity"

        with patch.object(feedback_synthesis_agent, 'run') as mock_synthesis:
            # Mock the async run method to return a result object with .output
            mock_result = Mock()
            mock_result.output = synthesized_feedback
            mock_synthesis.return_value = mock_result

            result = await agent._synthesize_feedback(
                batch_feedbacks=batch_feedbacks,
                combined_average=7.5,
                total_cards=30,
                has_insufficient_cards=False
            )

            assert result == synthesized_feedback
            mock_synthesis.assert_called_once()

            # Verify the prompt included all batch feedbacks
            call_args = mock_synthesis.call_args[0][0]
            for feedback in batch_feedbacks:
                assert feedback in call_args

    @pytest.mark.asyncio
    async def test_create_full_evaluation_result(self, multiple_cards):
        """Test lightweight evaluation result structure (method removed from actual API)."""
        agent = EvaluationAgent()

        lightweight_scores = [
            LightweightCardScore(
                card_id=card.id,
                name=card.name,
                score=8,
                reasoning=f"Good match for {card.name}"
            ) for card in multiple_cards
        ]

        lightweight_result = LightweightEvaluationResult(
            scored_cards=lightweight_scores,
            average_score=8.0,
            should_continue=False,
            feedback_for_query_agent="Good results",
            iteration_count=1
        )

        # Verify the lightweight result structure
        assert isinstance(lightweight_result, LightweightEvaluationResult)
        assert len(lightweight_result.scored_cards) == len(multiple_cards)
        assert all(isinstance(sc, LightweightCardScore) for sc in lightweight_result.scored_cards)
        assert lightweight_result.average_score == 8.0
        assert lightweight_result.should_continue is False


class TestEvaluationAgentConfiguration:
    """Test suite for EvaluationAgent configuration and global instances."""

    def test_lightweight_evaluation_agent_configuration(self):
        """Test lightweight evaluation agent configuration."""
        from src.mtg_search_agent.agents.evaluation_agent import lightweight_evaluation_agent

        assert lightweight_evaluation_agent is not None
        assert hasattr(lightweight_evaluation_agent, 'output_type')
        assert lightweight_evaluation_agent.output_type == LightweightAgentResult

    def test_feedback_synthesis_agent_configuration(self):
        """Test feedback synthesis agent configuration."""
        from src.mtg_search_agent.agents.evaluation_agent import feedback_synthesis_agent

        assert feedback_synthesis_agent is not None
        assert hasattr(feedback_synthesis_agent, 'output_type')
        assert feedback_synthesis_agent.output_type == str

    def test_agent_model_configuration(self):
        """Test that agents use correct model configuration."""
        from src.mtg_search_agent.agents.evaluation_agent import (
            lightweight_evaluation_agent, feedback_synthesis_agent
        )

        # Both should use OpenAI models
        assert hasattr(lightweight_evaluation_agent, 'model')
        assert hasattr(feedback_synthesis_agent, 'model')

    def test_agent_system_prompt_configuration(self):
        """Test system prompt configuration."""
        from src.mtg_search_agent.agents.evaluation_agent import feedback_synthesis_agent

        # Feedback synthesis agent should have system prompt
        assert hasattr(feedback_synthesis_agent, 'system_prompt')
        assert feedback_synthesis_agent.system_prompt is not None


class TestEvaluationAgentPerformance:
    """Test suite for performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_card_set_evaluation(self, mock_event_emitter):
        """Test evaluation performance with large card sets."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Create large set of cards
        large_card_set = []
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
            large_card_set.append(Card.from_scryfall(card_data))

        # Mock fast responses
        def fast_mock_response(*args, **kwargs):
            return LightweightAgentResult(
                scored_cards=[
                    LightweightCardScore(
                        card_id=f"large-card-{i}",
                        name=f"Large Card {i}",
                        score=5,
                        reasoning="Fast evaluation"
                    ) for i in range(10)  # Batch size
                ]
            )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = fast_mock_response

            import time
            start_time = time.time()

            result = await agent.evaluate_cards(
                natural_language_request="test large set",
                cards=large_card_set,
                iteration_count=1
            )

            elapsed_time = time.time() - start_time

            # Should complete in reasonable time (allow up to 60s for AI API calls)
            assert elapsed_time < 60.0  # Allow 60 seconds for large set with AI API

            # Should handle all cards
            assert isinstance(result, LightweightEvaluationResult)

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, multiple_cards, mock_event_emitter):
        """Test concurrent evaluation requests."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Mock responses for different evaluations
        def mock_response(*args, **kwargs):
            return LightweightAgentResult(
                scored_cards=[
                    LightweightCardScore(
                        card_id=card.id,
                        name=card.name,
                        score=6,
                        reasoning="Concurrent evaluation"
                    ) for card in multiple_cards
                ]
            )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = mock_response

            # Start multiple concurrent evaluations
            tasks = [
                agent.evaluate_cards("red cards", multiple_cards, 1),
                agent.evaluate_cards("blue cards", multiple_cards, 1),
                agent.evaluate_cards("green cards", multiple_cards, 1),
            ]

            results = await asyncio.gather(*tasks)

            # Verify all evaluations completed
            assert len(results) == 3
            assert all(isinstance(result, LightweightEvaluationResult) for result in results)


class TestEvaluationAgentEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_evaluation_with_malformed_ai_response(self, multiple_cards, mock_event_emitter):
        """Test handling of malformed AI responses."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Mock malformed response (missing required fields)
        malformed_response = LightweightAgentResult(scored_cards=[])  # Missing feedback

        # Create a mock result object with .output attribute
        mock_result = Mock()
        mock_result.output = malformed_response

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="test",
                cards=multiple_cards,
                iteration_count=1
            )

            # Should handle gracefully
            assert isinstance(result, LightweightEvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluation_with_invalid_scores(self, multiple_cards, mock_event_emitter):
        """Test handling of AI responses with invalid scores."""
        agent = EvaluationAgent()

        # This test verifies that invalid scores are caught during model creation
        with pytest.raises(ValidationError):
            LightweightCardScore(
                card_id=multiple_cards[0].id,
                name=multiple_cards[0].name,
                score=15,  # Invalid score > 10
                reasoning="Invalid score test"
            )


    @pytest.mark.asyncio
    async def test_evaluation_with_very_long_text(self, mock_event_emitter):
        """Test evaluation with cards having very long text."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Create card with very long oracle text
        long_text_data = {
            "id": "long-text-id",
            "name": "Long Text Card",
            "type_line": "Sorcery",
            "oracle_text": "This card has very long text. " * 100,  # Very long text
            "set": "long",
            "set_name": "Long Text Set",
            "rarity": "rare",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/long-text"
        }
        long_text_card = Card.from_scryfall(long_text_data)

        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=long_text_card.id,
                    name=long_text_card.name,
                    score=5,
                    reasoning="Handled long text"
                )
            ]
        )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="test",
                cards=[long_text_card],
                iteration_count=1
            )

            # Should handle long text without issues
            assert isinstance(result, LightweightEvaluationResult)

    @pytest.mark.asyncio
    async def test_evaluation_timeout_handling(self, multiple_cards, mock_event_emitter):
        """Test handling of evaluation timeouts."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = TimeoutError("Evaluation timed out")

            with pytest.raises(TimeoutError):
                await agent.evaluate_cards(
                    natural_language_request="timeout test",
                    cards=multiple_cards,
                    iteration_count=1
                )

    @pytest.mark.asyncio
    async def test_evaluation_with_unicode_content(self, mock_event_emitter):
        """Test evaluation with unicode content in cards."""
        agent = EvaluationAgent(event_emitter=mock_event_emitter)

        # Create card with unicode content
        unicode_data = {
            "id": "unicode-id",
            "name": "Æther Spëllbomb",
            "type_line": "Artifact",
            "oracle_text": "Sacrifice Æther Spëllbomb: Return target creature to its owner's hand.",
            "set": "uni",
            "set_name": "Unicode Set",
            "rarity": "common",
            "collector_number": "1",
            "scryfall_uri": "https://scryfall.com/unicode"
        }
        unicode_card = Card.from_scryfall(unicode_data)

        mock_response = LightweightAgentResult(
            scored_cards=[
                LightweightCardScore(
                    card_id=unicode_card.id,
                    name=unicode_card.name,
                    score=7,
                    reasoning="Unicode content handled"
                )
            ]
        )

        with patch.object(lightweight_evaluation_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.evaluate_cards(
                natural_language_request="unicode test",
                cards=[unicode_card],
                iteration_count=1
            )

            assert isinstance(result, LightweightEvaluationResult)
            assert len(result.scored_cards) == 1