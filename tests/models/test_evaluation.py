"""
Unit tests for Evaluation models.

Tests cover:
- BaseCardScore validation with score constraints
- CardScore with full card data
- LightweightCardScore with minimal data
- EvaluationResult and LightweightEvaluationResult
- Generics and type safety
- Edge cases and validation scenarios
"""
import pytest
from pydantic import ValidationError
from src.mtg_search_agent.models.evaluation import (
    BaseCardScore, CardScore, LightweightCardScore,
    EvaluationResult, LightweightEvaluationResult,
    LightweightAgentResult
)


class TestBaseCardScore:
    """Test suite for the BaseCardScore model."""

    def test_base_card_score_valid_scores(self):
        """Test BaseCardScore with valid score range (1-10)."""
        for score in range(1, 11):
            card_score = BaseCardScore(score=score)
            assert card_score.score == score
            assert card_score.reasoning is None

    def test_base_card_score_with_reasoning(self):
        """Test BaseCardScore with reasoning text."""
        reasoning = "This card is highly relevant to the search query."
        card_score = BaseCardScore(score=8, reasoning=reasoning)

        assert card_score.score == 8
        assert card_score.reasoning == reasoning

    def test_base_card_score_invalid_low_score(self):
        """Test BaseCardScore rejects scores below 1."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCardScore(score=0)

        error = exc_info.value
        assert "greater than or equal to 1" in str(error)

    def test_base_card_score_invalid_high_score(self):
        """Test BaseCardScore rejects scores above 10."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCardScore(score=11)

        error = exc_info.value
        assert "less than or equal to 10" in str(error)

    def test_base_card_score_negative_score(self):
        """Test BaseCardScore rejects negative scores."""
        with pytest.raises(ValidationError):
            BaseCardScore(score=-1)

    def test_base_card_score_float_score(self):
        """Test BaseCardScore with float scores within range."""
        with pytest.raises(ValidationError):
            # Score field should be int, not float based on the Field constraint
            BaseCardScore(score=5.5)

    def test_base_card_score_missing_score(self):
        """Test BaseCardScore requires score field."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCardScore(reasoning="Good card")

        error = exc_info.value
        assert "score" in str(error)

    def test_base_card_score_empty_reasoning(self):
        """Test BaseCardScore with empty reasoning string."""
        card_score = BaseCardScore(score=5, reasoning="")
        assert card_score.reasoning == ""

    def test_base_card_score_long_reasoning(self):
        """Test BaseCardScore with very long reasoning text."""
        long_reasoning = "This card is relevant. " * 100
        card_score = BaseCardScore(score=7, reasoning=long_reasoning)

        assert card_score.score == 7
        assert len(card_score.reasoning) > 1000


class TestCardScore:
    """Test suite for the CardScore model."""

    def test_card_score_creation(self, sample_card):
        """Test creating CardScore with Card instance."""
        card_score = CardScore(
            card=sample_card,
            score=8,
            reasoning="Lightning Bolt is a classic direct damage spell."
        )

        assert card_score.card == sample_card
        assert card_score.score == 8
        assert "Lightning Bolt" in card_score.reasoning

    def test_card_score_inherits_base_validation(self, sample_card):
        """Test CardScore inherits BaseCardScore validation."""
        # Should reject invalid scores
        with pytest.raises(ValidationError):
            CardScore(card=sample_card, score=0)

        with pytest.raises(ValidationError):
            CardScore(card=sample_card, score=15)

    def test_card_score_missing_card(self):
        """Test CardScore requires card field."""
        with pytest.raises(ValidationError) as exc_info:
            CardScore(score=5)

        error = exc_info.value
        assert "card" in str(error)

    def test_card_score_with_creature(self, sample_creature):
        """Test CardScore with creature card."""
        card_score = CardScore(
            card=sample_creature,
            score=6,
            reasoning="Serra Angel is a solid creature but not what we're looking for."
        )

        assert card_score.card.power == "4"
        assert card_score.card.toughness == "4"
        assert "creature" in card_score.reasoning.lower()

    def test_card_score_minimal_reasoning(self, sample_card):
        """Test CardScore with minimal reasoning."""
        card_score = CardScore(
            card=sample_card,
            score=9,
            reasoning="Perfect match."
        )

        assert card_score.reasoning == "Perfect match."

    def test_card_score_no_reasoning(self, sample_card):
        """Test CardScore without reasoning (should default to None)."""
        card_score = CardScore(card=sample_card, score=5)
        assert card_score.reasoning is None


class TestLightweightCardScore:
    """Test suite for the LightweightCardScore model."""

    def test_lightweight_card_score_creation(self):
        """Test creating LightweightCardScore with minimal data."""
        card_score = LightweightCardScore(
            card_id="test-id-123",
            name="Test Card",
            score=7,
            reasoning="Good match for the query."
        )

        assert card_score.card_id == "test-id-123"
        assert card_score.name == "Test Card"
        assert card_score.score == 7
        assert card_score.reasoning == "Good match for the query."

    def test_lightweight_card_score_inherits_validation(self):
        """Test LightweightCardScore inherits BaseCardScore validation."""
        with pytest.raises(ValidationError):
            LightweightCardScore(
                card_id="test-id",
                name="Test Card",
                score=0  # Invalid score
            )

    def test_lightweight_card_score_missing_fields(self):
        """Test LightweightCardScore requires card_id and name."""
        with pytest.raises(ValidationError):
            LightweightCardScore(score=5)  # Missing card_id and name

        with pytest.raises(ValidationError):
            LightweightCardScore(card_id="test-id", score=5)  # Missing name

        with pytest.raises(ValidationError):
            LightweightCardScore(name="Test Card", score=5)  # Missing card_id

    def test_lightweight_card_score_empty_strings(self):
        """Test LightweightCardScore with empty strings."""
        with pytest.raises(ValidationError):
            LightweightCardScore(
                card_id="",  # Empty string should be invalid
                name="Test Card",
                score=5
            )

        with pytest.raises(ValidationError):
            LightweightCardScore(
                card_id="test-id",
                name="",  # Empty string should be invalid
                score=5
            )

    def test_lightweight_card_score_unicode_names(self):
        """Test LightweightCardScore with unicode card names."""
        card_score = LightweightCardScore(
            card_id="unicode-id",
            name="Æther Spëllbomb",
            score=6,
            reasoning="Card with special characters."
        )

        assert "Æther" in card_score.name
        assert "special characters" in card_score.reasoning

    def test_lightweight_card_score_long_names(self):
        """Test LightweightCardScore with very long card names."""
        long_name = "This Is A Very Long Card Name That Exceeds Normal Limits " * 3
        card_score = LightweightCardScore(
            card_id="long-name-id",
            name=long_name,
            score=4
        )

        assert len(card_score.name) > 100


class TestEvaluationResult:
    """Test suite for the EvaluationResult model."""

    def test_evaluation_result_creation(self, sample_card_score, sample_creature):
        """Test creating EvaluationResult with scored cards."""
        creature_score = CardScore(card=sample_creature, score=6)
        scored_cards = [sample_card_score, creature_score]
        average_score = (sample_card_score.score + creature_score.score) / 2

        result = EvaluationResult(
            scored_cards=scored_cards,
            average_score=average_score,
            should_continue=False,
            feedback_for_query_agent="Good results found.",
            iteration_count=1
        )

        assert len(result.scored_cards) == 2
        assert result.average_score == 7.0  # (8 + 6) / 2
        assert result.should_continue is False
        assert result.feedback_for_query_agent == "Good results found."
        assert result.iteration_count == 1

    def test_evaluation_result_empty_scored_cards(self):
        """Test EvaluationResult with empty scored cards list."""
        result = EvaluationResult(
            scored_cards=[],
            average_score=0.0,
            should_continue=True,
            feedback_for_query_agent="No cards found, try different query.",
            iteration_count=1
        )

        assert result.scored_cards == []
        assert result.average_score == 0.0
        assert result.should_continue is True

    def test_evaluation_result_single_card(self, sample_card_score):
        """Test EvaluationResult with single scored card."""
        result = EvaluationResult(
            scored_cards=[sample_card_score],
            average_score=float(sample_card_score.score),
            should_continue=False,
            iteration_count=1
        )

        assert len(result.scored_cards) == 1
        assert result.average_score == 8.0
        assert result.feedback_for_query_agent is None  # Optional field

    def test_evaluation_result_high_iteration_count(self, sample_card_score):
        """Test EvaluationResult with high iteration count."""
        result = EvaluationResult(
            scored_cards=[sample_card_score],
            average_score=8.0,
            should_continue=False,
            iteration_count=5
        )

        assert result.iteration_count == 5

    def test_evaluation_result_missing_required_fields(self, sample_card_score):
        """Test EvaluationResult validation with missing fields."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                scored_cards=[sample_card_score],
                average_score=8.0
                # Missing should_continue and iteration_count
            )

    def test_evaluation_result_negative_average_score(self, sample_card_score):
        """Test EvaluationResult with negative average score."""
        # This might be valid in some edge cases
        result = EvaluationResult(
            scored_cards=[sample_card_score],
            average_score=-1.0,
            should_continue=True,
            iteration_count=1
        )

        assert result.average_score == -1.0

    def test_evaluation_result_very_high_average_score(self, sample_card_score):
        """Test EvaluationResult with average score above 10."""
        # This should be allowed as it's computed from individual scores
        result = EvaluationResult(
            scored_cards=[sample_card_score],
            average_score=15.0,
            should_continue=False,
            iteration_count=1
        )

        assert result.average_score == 15.0


class TestLightweightEvaluationResult:
    """Test suite for the LightweightEvaluationResult model."""

    def test_lightweight_evaluation_result_creation(self, sample_lightweight_card_score):
        """Test creating LightweightEvaluationResult."""
        scored_cards = [
            sample_lightweight_card_score,
            LightweightCardScore(
                card_id="second-id",
                name="Second Card",
                score=6
            )
        ]

        result = LightweightEvaluationResult(
            scored_cards=scored_cards,
            average_score=7.0,
            should_continue=False,
            feedback_for_query_agent="Lightweight evaluation complete.",
            iteration_count=2
        )

        assert len(result.scored_cards) == 2
        assert result.average_score == 7.0
        assert result.iteration_count == 2
        assert "Lightweight" in result.feedback_for_query_agent

    def test_lightweight_evaluation_result_empty_cards(self):
        """Test LightweightEvaluationResult with no cards."""
        result = LightweightEvaluationResult(
            scored_cards=[],
            average_score=0.0,
            should_continue=True,
            feedback_for_query_agent="No cards to evaluate.",
            iteration_count=1
        )

        assert len(result.scored_cards) == 0
        assert result.should_continue is True

    def test_lightweight_evaluation_result_many_cards(self):
        """Test LightweightEvaluationResult with many lightweight cards."""
        scored_cards = []
        for i in range(50):
            scored_cards.append(
                LightweightCardScore(
                    card_id=f"card-id-{i}",
                    name=f"Card {i}",
                    score=(i % 10) + 1  # Scores 1-10
                )
            )

        result = LightweightEvaluationResult(
            scored_cards=scored_cards,
            average_score=5.5,
            should_continue=False,
            iteration_count=3
        )

        assert len(result.scored_cards) == 50
        assert result.average_score == 5.5


class TestLightweightAgentResult:
    """Test suite for the deprecated LightweightAgentResult model."""

    def test_lightweight_agent_result_creation(self, sample_lightweight_card_score):
        """Test creating LightweightAgentResult (backward compatibility)."""
        scored_cards = [sample_lightweight_card_score]

        result = LightweightAgentResult(
            scored_cards=scored_cards,
            feedback_for_query_agent="Agent feedback."
        )

        assert len(result.scored_cards) == 1
        assert result.feedback_for_query_agent == "Agent feedback."

    def test_lightweight_agent_result_minimal(self):
        """Test LightweightAgentResult with minimal required fields."""
        card_score = LightweightCardScore(
            card_id="minimal-id",
            name="Minimal Card",
            score=5
        )

        result = LightweightAgentResult(scored_cards=[card_score])

        assert len(result.scored_cards) == 1
        assert result.feedback_for_query_agent is None

    def test_lightweight_agent_result_empty_cards(self):
        """Test LightweightAgentResult with empty cards list."""
        result = LightweightAgentResult(
            scored_cards=[],
            feedback_for_query_agent="No results found."
        )

        assert result.scored_cards == []
        assert result.feedback_for_query_agent == "No results found."


class TestEvaluationModelsIntegration:
    """Integration tests for evaluation models working together."""

    def test_converting_lightweight_to_full_evaluation(
        self, sample_lightweight_evaluation_result, sample_card, sample_creature
    ):
        """Test pattern of converting lightweight results to full results."""
        lightweight_result = sample_lightweight_evaluation_result

        # Simulate converting lightweight scores to full CardScore objects
        # This would typically happen when we have the full card data
        full_scores = []
        for lightweight_score in lightweight_result.scored_cards:
            if lightweight_score.name == "Lightning Bolt":
                card = sample_card
            else:
                card = sample_creature

            full_score = CardScore(
                card=card,
                score=lightweight_score.score,
                reasoning=lightweight_score.reasoning
            )
            full_scores.append(full_score)

        full_result = EvaluationResult(
            scored_cards=full_scores,
            average_score=lightweight_result.average_score,
            should_continue=lightweight_result.should_continue,
            feedback_for_query_agent=lightweight_result.feedback_for_query_agent,
            iteration_count=lightweight_result.iteration_count
        )

        assert len(full_result.scored_cards) == len(lightweight_result.scored_cards)
        assert full_result.average_score == lightweight_result.average_score
        assert full_result.should_continue == lightweight_result.should_continue

    def test_score_consistency_across_models(self, sample_card):
        """Test that score validation is consistent across all score models."""
        # All should reject scores outside 1-10 range
        invalid_scores = [0, -1, 11, 15]

        for score in invalid_scores:
            with pytest.raises(ValidationError):
                BaseCardScore(score=score)

            with pytest.raises(ValidationError):
                CardScore(card=sample_card, score=score)

            with pytest.raises(ValidationError):
                LightweightCardScore(
                    card_id="test-id",
                    name="Test Card",
                    score=score
                )

    def test_evaluation_results_with_mixed_scores(self, sample_card, sample_creature):
        """Test evaluation results with cards having different score ranges."""
        scored_cards = [
            CardScore(card=sample_card, score=1, reasoning="Lowest score"),
            CardScore(card=sample_creature, score=10, reasoning="Highest score"),
            CardScore(card=sample_card, score=5, reasoning="Middle score"),
        ]

        average = sum(sc.score for sc in scored_cards) / len(scored_cards)

        result = EvaluationResult(
            scored_cards=scored_cards,
            average_score=average,
            should_continue=True,
            iteration_count=1
        )

        assert result.average_score == pytest.approx(5.33, rel=1e-2)
        assert min(sc.score for sc in result.scored_cards) == 1
        assert max(sc.score for sc in result.scored_cards) == 10