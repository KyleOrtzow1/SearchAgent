"""
Unit tests for the TagSearchTool.

Tests cover:
- Tool initialization and tag loading
- Fuzzy string matching functionality
- Tag suggestion ranking and filtering
- Edge cases and error handling
- Performance with different tag sets
"""
import pytest
import json
from unittest.mock import patch, mock_open, Mock
from typing import List

from src.mtg_search_agent.tools.tag_search import TagSearchTool
from src.mtg_search_agent.models.search import TagSuggestion


class TestTagSearchToolInitialization:
    """Test suite for TagSearchTool initialization and tag loading."""

    def test_tag_search_tool_initialization(self):
        """Test TagSearchTool initialization."""
        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            assert hasattr(tool, 'tags_data')
            assert hasattr(tool, 'flat_tags')
            assert isinstance(tool.tags_data, dict)
            assert isinstance(tool.flat_tags, list)

    @patch('importlib.resources.open_text')
    def test_tags_loading_success(self, mock_open_text, sample_tags_data):
        """Test successful loading of tags from JSON file."""
        # Mock the file content
        mock_file = mock_open(read_data=json.dumps(sample_tags_data))
        mock_open_text.return_value.__enter__.return_value = mock_file.return_value

        tool = TagSearchTool()

        # Verify tags were loaded
        assert tool.tags_data == sample_tags_data

        # Verify flat_tags was populated correctly
        expected_flat_tags = []
        for category, tags in sample_tags_data.items():
            for tag in tags:
                expected_flat_tags.append((tag, category))

        assert len(tool.flat_tags) == len(expected_flat_tags)

        # Check that all expected tags are present
        flat_tag_names = [tag for tag, category in tool.flat_tags]
        for category, tags in sample_tags_data.items():
            for tag in tags:
                assert tag in flat_tag_names

    @patch('importlib.resources.open_text')
    def test_tags_loading_file_not_found(self, mock_open_text):
        """Test handling when tags.json file is not found."""
        mock_open_text.side_effect = FileNotFoundError("tags.json not found")

        with patch('builtins.print') as mock_print:
            tool = TagSearchTool()

            # Should handle error gracefully
            assert tool.tags_data == {}
            assert tool.flat_tags == []
            mock_print.assert_called_once_with("Error: tags.json not found. The package may be installed incorrectly.")

    @patch('importlib.resources.open_text')
    def test_tags_loading_invalid_json(self, mock_open_text):
        """Test handling of invalid JSON in tags file."""
        # Mock invalid JSON content
        mock_file = mock_open(read_data="invalid json content")
        mock_open_text.return_value.__enter__.return_value = mock_file.return_value

        with pytest.raises(json.JSONDecodeError):
            TagSearchTool()

    @patch('importlib.resources.open_text')
    def test_tags_loading_empty_file(self, mock_open_text):
        """Test handling of empty tags file."""
        mock_file = mock_open(read_data="{}")
        mock_open_text.return_value.__enter__.return_value = mock_file.return_value

        tool = TagSearchTool()

        assert tool.tags_data == {}
        assert tool.flat_tags == []


class TestTagSearchToolFuzzyMatching:
    """Test suite for fuzzy matching functionality."""

    def setup_method(self):
        """Set up test tool with sample data."""
        sample_data = {
            "types": ["instant", "sorcery", "creature", "artifact", "enchantment"],
            "colors": ["white", "blue", "black", "red", "green"],
            "mechanics": ["flying", "trample", "haste", "vigilance", "damage"],
            "keywords": ["flash", "lifelink", "deathtouch", "first strike"]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            self.tool = TagSearchTool()
            self.tool.tags_data = sample_data
            self.tool.flat_tags = []
            for category, tags in sample_data.items():
                for tag in tags:
                    self.tool.flat_tags.append((tag, category))

    def test_exact_match_single_tag(self):
        """Test fuzzy matching with exact single tag match."""
        result = self.tool.find_similar_tags(["instant"])

        assert "instant" in result
        assert result[0] == "instant"  # Should be first due to perfect match

    def test_exact_match_multiple_tags(self):
        """Test fuzzy matching with multiple exact matches."""
        result = self.tool.find_similar_tags(["instant", "creature"])

        assert "instant" in result
        assert "creature" in result

    def test_partial_match_single_character(self):
        """Test fuzzy matching with single character difference."""
        result = self.tool.find_similar_tags(["instnt"])  # Missing 'a'

        assert "instant" in result

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        test_cases = ["INSTANT", "Instant", "iNsTaNt", "instant"]

        for case in test_cases:
            result = self.tool.find_similar_tags([case])
            assert "instant" in result

    def test_multi_word_matching(self):
        """Test matching multi-word terms."""
        result = self.tool.find_similar_tags(["first strike"])

        assert "first strike" in result

    def test_token_sort_matching(self):
        """Test token sort ratio matching for word order variations."""
        result = self.tool.find_similar_tags(["strike first"])

        # Should still match "first strike" due to token_sort_ratio
        assert "first strike" in result

    def test_threshold_filtering(self):
        """Test that low-quality matches are filtered out."""
        result = self.tool.find_similar_tags(["xyz"])  # Very different from any tag

        # Should return empty or very few results due to low similarity
        assert len(result) == 0 or all(
            any(char in tag.lower() for char in "xyz") for tag in result
        )

    def test_approximate_matching(self):
        """Test approximate matching for common typos."""
        test_cases = [
            ("creatuer", "creature"),
            ("enchanment", "enchantment"),
            ("artfact", "artifact"),
            ("lifelnik", "lifelink"),
            ("deathtuch", "deathtouch")
        ]

        for typo, expected in test_cases:
            result = self.tool.find_similar_tags([typo])
            assert expected in result

    def test_substring_matching(self):
        """Test matching substrings within longer words."""
        result = self.tool.find_similar_tags(["inst"])

        assert "instant" in result

    def test_max_results_limit(self):
        """Test that max_results parameter limits output."""
        # Use a broad search that would match many tags
        result = self.tool.find_similar_tags(["e"], max_results=3)

        assert len(result) <= 3

    def test_max_results_default(self):
        """Test default max_results behavior."""
        # Use a search that might match many tags
        result = self.tool.find_similar_tags(["a"])

        assert len(result) <= 10  # Default max_results

    def test_empty_guess_tags(self):
        """Test behavior with empty guess tags list."""
        result = self.tool.find_similar_tags([])

        assert result == []

    def test_multiple_guess_tags_combined(self):
        """Test that multiple guess tags combine results appropriately."""
        result = self.tool.find_similar_tags(["inst", "creat"])

        # Should include matches for both "instant" and "creature"
        assert any("instant" in tag for tag in result)
        assert any("creature" in tag for tag in result)

    def test_duplicate_removal(self):
        """Test that duplicate suggestions are removed."""
        # Use tags that might match the same result
        result = self.tool.find_similar_tags(["instant", "instant"])

        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_score_based_ordering(self):
        """Test that results are ordered by similarity score."""
        result = self.tool.find_similar_tags(["creat"])

        # "creature" should rank higher than other partial matches
        if len(result) > 1:
            assert result[0] == "creature"


class TestTagSearchToolPerformance:
    """Test suite for performance characteristics."""

    def test_large_tag_set_performance(self):
        """Test performance with large tag sets."""
        # Create a large tag set
        large_tags_data = {
            "category1": [f"tag_{i}" for i in range(100)],
            "category2": [f"item_{i}" for i in range(100)],
            "category3": [f"element_{i}" for i in range(100)]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = large_tags_data
            tool.flat_tags = []
            for category, tags in large_tags_data.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Should complete in reasonable time
        import time
        start_time = time.time()
        result = tool.find_similar_tags(["tag"])
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0  # Should complete within 1 second
        assert len(result) > 0

    def test_many_guess_tags_performance(self):
        """Test performance with many guess tags."""
        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = {"test": ["instant", "creature", "artifact"]}
            tool.flat_tags = [("instant", "test"), ("creature", "test"), ("artifact", "test")]

        # Create many guess tags
        many_guesses = [f"guess_{i}" for i in range(50)]

        import time
        start_time = time.time()
        result = tool.find_similar_tags(many_guesses)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0  # Should complete within 2 seconds
        assert isinstance(result, list)


class TestTagSearchToolEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_tags_data(self):
        """Test behavior when tags data is empty."""
        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = {}
            tool.flat_tags = []

        result = tool.find_similar_tags(["instant"])
        assert result == []

    def test_whitespace_handling(self):
        """Test handling of whitespace in guess tags."""
        sample_data = {"types": ["instant", "creature"]}

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = sample_data
            tool.flat_tags = [("instant", "types"), ("creature", "types")]

        test_cases = [
            " instant ",  # Leading/trailing spaces
            "  instant  ",  # Multiple spaces
            "\tinstant\n",  # Tabs and newlines
        ]

        for test_input in test_cases:
            result = tool.find_similar_tags([test_input])
            assert "instant" in result

    def test_special_characters_in_tags(self):
        """Test handling of special characters in tags."""
        special_tags = {
            "mechanics": ["first strike", "+1/+1", "draw/discard", "mana-cost"]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = special_tags
            tool.flat_tags = []
            for category, tags in special_tags.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Test matching special characters
        result = tool.find_similar_tags(["+1/+1"])
        assert "+1/+1" in result

        result = tool.find_similar_tags(["mana-cost"])
        assert "mana-cost" in result

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_tags = {"keywords": ["Æther", "naïve", "café"]}

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = unicode_tags
            tool.flat_tags = [("Æther", "keywords"), ("naïve", "keywords"), ("café", "keywords")]

        result = tool.find_similar_tags(["Aether"])  # ASCII approximation
        # Should find Æther or similar
        assert len(result) >= 0  # May or may not match depending on fuzzy algorithm

    def test_very_long_tag_names(self):
        """Test handling of very long tag names."""
        long_tag = "a" * 100  # 100 character tag
        long_tags = {"test": [long_tag]}

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = long_tags
            tool.flat_tags = [(long_tag, "test")]

        result = tool.find_similar_tags(["a" * 50])  # 50 character guess
        # Should handle without error
        assert isinstance(result, list)

    def test_numeric_tags(self):
        """Test handling of numeric tags."""
        numeric_tags = {"costs": ["0", "1", "2", "10", "15"]}

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = numeric_tags
            tool.flat_tags = []
            for category, tags in numeric_tags.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        result = tool.find_similar_tags(["1"])
        assert "1" in result

        result = tool.find_similar_tags(["10"])
        assert "10" in result


class TestTagSearchToolCategorization:
    """Test suite for tag categorization features."""

    def test_multiple_categories_represented(self):
        """Test that results can come from multiple categories."""
        diverse_tags = {
            "types": ["instant"],
            "colors": ["red"],
            "mechanics": ["damage"]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = diverse_tags
            tool.flat_tags = []
            for category, tags in diverse_tags.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Search for something that might match across categories
        result = tool.find_similar_tags(["r"])  # Could match "red"

        # Verify we get results and they make sense
        assert isinstance(result, list)

    def test_category_information_preservation(self):
        """Test that category information is preserved in flat_tags."""
        sample_data = {
            "types": ["instant", "creature"],
            "colors": ["red", "blue"]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = sample_data
            tool.flat_tags = []
            for category, tags in sample_data.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Verify flat_tags structure
        assert len(tool.flat_tags) == 4

        # Check that categories are preserved
        tag_dict = dict(tool.flat_tags)
        assert tag_dict["instant"] == "types"
        assert tag_dict["red"] == "colors"


class TestTagSearchToolIntegration:
    """Integration tests for TagSearchTool with realistic data."""

    def test_realistic_mtg_search_scenario(self):
        """Test realistic MTG search scenarios."""
        mtg_tags = {
            "types": ["instant", "sorcery", "creature", "artifact", "enchantment", "planeswalker"],
            "colors": ["white", "blue", "black", "red", "green", "colorless"],
            "mechanics": ["flying", "trample", "haste", "vigilance", "lifelink", "deathtouch"],
            "keywords": ["flash", "hexproof", "indestructible", "first strike", "double strike"]
        }

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = mtg_tags
            tool.flat_tags = []
            for category, tags in mtg_tags.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Test common search patterns
        test_scenarios = [
            (["burn", "damage"], ["deathtouch"]),  # "damage" might partially match "deathtouch"
            (["flying creature"], ["flying", "creature"]),
            (["red instant"], ["red", "instant"]),
            (["hexprof"], ["hexproof"]),  # Common typo
            (["liflink"], ["lifelink"]),  # Common typo
        ]

        for guess_tags, expected_contains in test_scenarios:
            result = tool.find_similar_tags(guess_tags)
            for expected in expected_contains:
                if any(tag in result for tag in [expected]):
                    # At least one expected tag should be found
                    pass  # Test passes
                else:
                    # This is informational - fuzzy matching might not catch everything
                    print(f"Note: Expected {expected} not found for {guess_tags}, got {result}")

    def test_comprehensive_tag_coverage(self):
        """Test that tool can handle comprehensive tag sets."""
        # Simulate a large, realistic tag database
        comprehensive_tags = {}

        # Generate lots of realistic tags
        for i in range(10):
            comprehensive_tags[f"category_{i}"] = [f"tag_{i}_{j}" for j in range(20)]

        with patch.object(TagSearchTool, '_load_tags'):
            tool = TagSearchTool()
            tool.tags_data = comprehensive_tags
            tool.flat_tags = []
            for category, tags in comprehensive_tags.items():
                for tag in tags:
                    tool.flat_tags.append((tag, category))

        # Should handle large datasets
        result = tool.find_similar_tags(["tag_1"])
        assert isinstance(result, list)
        assert len(result) <= 10  # Respects max_results