"""
Unit tests for Search models.

Tests cover:
- SearchQuery model validation and behavior
- SearchResult model with cards and pagination
- TagSuggestion model for fuzzy matching results
- Edge cases and validation scenarios
"""
import pytest
from pydantic import ValidationError
from src.mtg_search_agent.models.search import SearchQuery, SearchResult, TagSuggestion


class TestSearchQuery:
    """Test suite for the SearchQuery model."""

    def test_search_query_basic_creation(self):
        """Test creating a basic SearchQuery."""
        query = SearchQuery(query="c:r t:instant")

        assert query.query == "c:r t:instant"
        assert query.explanation is None
        assert query.confidence is None

    def test_search_query_with_all_fields(self):
        """Test creating SearchQuery with all fields."""
        query = SearchQuery(
            query="c:r t:instant",
            explanation="Search for red instant spells",
            confidence=0.85
        )

        assert query.query == "c:r t:instant"
        assert query.explanation == "Search for red instant spells"
        assert query.confidence == 0.85

    def test_search_query_empty_query_string(self):
        """Test SearchQuery with empty query string."""
        query = SearchQuery(query="")
        assert query.query == ""

    def test_search_query_complex_scryfall_syntax(self):
        """Test SearchQuery with complex Scryfall syntax."""
        complex_query = "c:r t:instant cmc>=3 (o:damage or o:burn) -o:target"
        query = SearchQuery(
            query=complex_query,
            explanation="Red instants with CMC 3+ that deal damage but don't target",
            confidence=0.75
        )

        assert query.query == complex_query
        assert "damage" in query.explanation
        assert query.confidence == 0.75

    def test_search_query_confidence_bounds(self):
        """Test SearchQuery confidence value bounds."""
        # Valid confidence values
        query1 = SearchQuery(query="test", confidence=0.0)
        assert query1.confidence == 0.0

        query2 = SearchQuery(query="test", confidence=1.0)
        assert query2.confidence == 1.0

        query3 = SearchQuery(query="test", confidence=0.5)
        assert query3.confidence == 0.5

    def test_search_query_invalid_confidence_types(self):
        """Test SearchQuery with invalid confidence types."""
        with pytest.raises(ValidationError):
            SearchQuery(query="test", confidence="high")

        with pytest.raises(ValidationError):
            SearchQuery(query="test", confidence=[0.5])

    def test_search_query_missing_required_field(self):
        """Test SearchQuery validation when missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            SearchQuery()

        error = exc_info.value
        assert "query" in str(error)

    def test_search_query_unicode_characters(self):
        """Test SearchQuery with unicode characters."""
        query = SearchQuery(
            query="o:\"Æther Vial\"",
            explanation="Search for cards mentioning Æther Vial"
        )

        assert "Æther" in query.query
        assert "Æther" in query.explanation

    def test_search_query_very_long_strings(self):
        """Test SearchQuery with very long strings."""
        long_query = "c:r " * 100  # Very long query
        long_explanation = "This is a very long explanation. " * 50

        query = SearchQuery(
            query=long_query.strip(),
            explanation=long_explanation.strip(),
            confidence=0.1
        )

        assert len(query.query) > 200
        assert len(query.explanation) > 1000


class TestSearchResult:
    """Test suite for the SearchResult model."""

    def test_search_result_basic_creation(self, sample_search_query_model, multiple_cards):
        """Test creating a basic SearchResult."""
        result = SearchResult(
            query=sample_search_query_model,
            cards=multiple_cards,
            total_cards=len(multiple_cards)
        )

        assert result.query == sample_search_query_model
        assert result.cards == multiple_cards
        assert result.total_cards == len(multiple_cards)
        assert result.has_more is False  # Default value

    def test_search_result_with_pagination(self, sample_search_query_model, multiple_cards):
        """Test SearchResult with pagination indicators."""
        result = SearchResult(
            query=sample_search_query_model,
            cards=multiple_cards,
            total_cards=175,  # More cards available than returned
            has_more=True
        )

        assert result.total_cards == 175
        assert result.has_more is True
        assert len(result.cards) < result.total_cards

    def test_search_result_empty_cards(self, sample_search_query_model):
        """Test SearchResult with no cards found."""
        result = SearchResult(
            query=sample_search_query_model,
            cards=[],
            total_cards=0,
            has_more=False
        )

        assert result.cards == []
        assert result.total_cards == 0
        assert result.has_more is False

    def test_search_result_single_card(self, sample_search_query_model, sample_card):
        """Test SearchResult with single card."""
        result = SearchResult(
            query=sample_search_query_model,
            cards=[sample_card],
            total_cards=1,
            has_more=False
        )

        assert len(result.cards) == 1
        assert result.cards[0] == sample_card
        assert result.total_cards == 1

    def test_search_result_large_result_set(self, sample_search_query_model, sample_card):
        """Test SearchResult with large numbers."""
        # Create a result representing a large search
        large_cards = [sample_card] * 500  # Simulate 500 cards returned
        result = SearchResult(
            query=sample_search_query_model,
            cards=large_cards,
            total_cards=5000,  # Total available
            has_more=True
        )

        assert len(result.cards) == 500
        assert result.total_cards == 5000
        assert result.has_more is True

    def test_search_result_missing_required_fields(self, sample_search_query_model):
        """Test SearchResult validation with missing fields."""
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(query=sample_search_query_model)

        error = exc_info.value
        assert "cards" in str(error) or "total_cards" in str(error)

    def test_search_result_invalid_total_cards(self, sample_search_query_model, multiple_cards):
        """Test SearchResult with invalid total_cards."""
        with pytest.raises(ValidationError):
            SearchResult(
                query=sample_search_query_model,
                cards=multiple_cards,
                total_cards="not_a_number"
            )

    def test_search_result_negative_total_cards(self, sample_search_query_model, multiple_cards):
        """Test SearchResult with negative total_cards."""
        with pytest.raises(ValidationError):
            SearchResult(
                query=sample_search_query_model,
                cards=multiple_cards,
                total_cards=-1
            )

    def test_search_result_cards_count_consistency(self, sample_search_query_model, multiple_cards):
        """Test relationship between cards list and total_cards."""
        # This is more of a logical test - the model doesn't enforce this constraint
        # but it documents expected behavior
        result = SearchResult(
            query=sample_search_query_model,
            cards=multiple_cards,
            total_cards=len(multiple_cards),
            has_more=False
        )

        # When has_more is False, cards length should equal total_cards
        assert len(result.cards) == result.total_cards
        assert result.has_more is False

    def test_search_result_pagination_logic(self, sample_search_query_model, multiple_cards):
        """Test pagination logic indicators."""
        # Case 1: Partial results (has_more = True)
        result1 = SearchResult(
            query=sample_search_query_model,
            cards=multiple_cards,
            total_cards=100,  # More than cards returned
            has_more=True
        )
        assert len(result1.cards) < result1.total_cards

        # Case 2: Complete results (has_more = False)
        result2 = SearchResult(
            query=sample_search_query_model,
            cards=multiple_cards,
            total_cards=len(multiple_cards),
            has_more=False
        )
        assert len(result2.cards) == result2.total_cards


class TestTagSuggestion:
    """Test suite for the TagSuggestion model."""

    def test_tag_suggestion_basic_creation(self):
        """Test creating a basic TagSuggestion."""
        suggestion = TagSuggestion(
            tag="instant",
            score=0.95,
            category="type"
        )

        assert suggestion.tag == "instant"
        assert suggestion.score == 0.95
        assert suggestion.category == "type"

    def test_tag_suggestion_different_categories(self):
        """Test TagSuggestion with different category types."""
        categories = ["type", "color", "mechanic", "keyword", "ability"]

        for category in categories:
            suggestion = TagSuggestion(
                tag=f"test_{category}",
                score=0.8,
                category=category
            )
            assert suggestion.category == category

    def test_tag_suggestion_score_bounds(self):
        """Test TagSuggestion with various score values."""
        # Test edge cases for score values
        test_scores = [0.0, 0.01, 0.5, 0.99, 1.0]

        for score in test_scores:
            suggestion = TagSuggestion(
                tag="test",
                score=score,
                category="test"
            )
            assert suggestion.score == score

    def test_tag_suggestion_negative_score(self):
        """Test TagSuggestion with negative score (should be allowed for fuzzy matching)."""
        suggestion = TagSuggestion(
            tag="test",
            score=-0.1,  # Negative scores might be valid for some fuzzy algorithms
            category="test"
        )
        assert suggestion.score == -0.1

    def test_tag_suggestion_high_score(self):
        """Test TagSuggestion with score above 1.0."""
        suggestion = TagSuggestion(
            tag="exact_match",
            score=1.5,  # Some fuzzy algorithms might return > 1.0
            category="type"
        )
        assert suggestion.score == 1.5

    def test_tag_suggestion_empty_tag(self):
        """Test TagSuggestion with empty tag string."""
        suggestion = TagSuggestion(
            tag="",
            score=0.5,
            category="type"
        )
        assert suggestion.tag == ""

    def test_tag_suggestion_unicode_tag(self):
        """Test TagSuggestion with unicode characters in tag."""
        suggestion = TagSuggestion(
            tag="Æther",
            score=0.9,
            category="keyword"
        )
        assert suggestion.tag == "Æther"

    def test_tag_suggestion_special_characters(self):
        """Test TagSuggestion with special characters."""
        special_tags = [
            "first strike",  # Space
            "mana-cost",     # Hyphen
            "draw/discard",  # Slash
            "x=0",           # Equals
            "+1/+1",         # Plus/minus
        ]

        for tag in special_tags:
            suggestion = TagSuggestion(
                tag=tag,
                score=0.8,
                category="mechanic"
            )
            assert suggestion.tag == tag

    def test_tag_suggestion_missing_required_fields(self):
        """Test TagSuggestion validation with missing fields."""
        with pytest.raises(ValidationError):
            TagSuggestion(tag="test", score=0.5)  # Missing category

        with pytest.raises(ValidationError):
            TagSuggestion(tag="test", category="type")  # Missing score

        with pytest.raises(ValidationError):
            TagSuggestion(score=0.5, category="type")  # Missing tag

    def test_tag_suggestion_invalid_score_type(self):
        """Test TagSuggestion with invalid score type."""
        with pytest.raises(ValidationError):
            TagSuggestion(
                tag="test",
                score="high",  # Should be numeric
                category="type"
            )

    def test_tag_suggestion_ordering(self):
        """Test TagSuggestion comparison for ordering by score."""
        suggestions = [
            TagSuggestion(tag="low", score=0.3, category="type"),
            TagSuggestion(tag="high", score=0.9, category="type"),
            TagSuggestion(tag="medium", score=0.6, category="type"),
        ]

        # Sort by score descending
        sorted_suggestions = sorted(suggestions, key=lambda x: x.score, reverse=True)

        assert sorted_suggestions[0].tag == "high"
        assert sorted_suggestions[1].tag == "medium"
        assert sorted_suggestions[2].tag == "low"

    def test_tag_suggestion_equality(self):
        """Test TagSuggestion equality comparison."""
        suggestion1 = TagSuggestion(tag="instant", score=0.95, category="type")
        suggestion2 = TagSuggestion(tag="instant", score=0.95, category="type")
        suggestion3 = TagSuggestion(tag="sorcery", score=0.95, category="type")

        assert suggestion1 == suggestion2
        assert suggestion1 != suggestion3

    def test_tag_suggestion_list_operations(self):
        """Test TagSuggestion in list operations."""
        suggestions = [
            TagSuggestion(tag="instant", score=0.95, category="type"),
            TagSuggestion(tag="damage", score=0.85, category="mechanic"),
            TagSuggestion(tag="red", score=0.90, category="color")
        ]

        # Test filtering
        type_suggestions = [s for s in suggestions if s.category == "type"]
        assert len(type_suggestions) == 1
        assert type_suggestions[0].tag == "instant"

        # Test finding max score
        max_score = max(s.score for s in suggestions)
        assert max_score == 0.95

        # Test finding by tag
        damage_suggestions = [s for s in suggestions if "damage" in s.tag]
        assert len(damage_suggestions) == 1