import json
import importlib.resources
from typing import List, Dict, Any
from fuzzywuzzy import fuzz

from ..models.search import TagSuggestion


class TagSearchTool:
    """Tool for finding similar tags using fuzzy string matching"""
    
    def __init__(self):
        self.tags_data: Dict[str, List[str]] = {}
        self.flat_tags: List[tuple] = []  # (tag, category)
        self._load_tags()
    
    def _load_tags(self) -> None:
        """Load tags from the JSON file included with the package"""
        try:
            # This new method reliably finds the data file within the package
            with importlib.resources.open_text('mtg_search_agent', 'tags.json') as f:
                self.tags_data = json.load(f)
        except FileNotFoundError:
            print("Error: tags.json not found. The package may be installed incorrectly.")
            return

        # Create flat list of (tag, category) tuples for efficient searching
        for category, tags in self.tags_data.items():
            for tag in tags:
                self.flat_tags.append((tag, category))
    
    def find_similar_tags(self, guess_tags: List[str], max_results: int = 10) -> List[str]:
        """
        Find tags similar to the provided guesses using fuzzy matching
        
        Args:
            guess_tags: List of tag guesses from the agent
            max_results: Maximum number of suggestions to return
            
        Returns:
            List of tag names (strings only) sorted by relevance score
        """
        tag_scores = []
        
        for guess in guess_tags:
            # Calculate fuzzy match scores for all tags
            for tag, category in self.flat_tags:
                # Use token_sort_ratio for better matching of multi-word terms
                score = fuzz.token_sort_ratio(guess.lower(), tag.lower())
                
                # Only include matches above a threshold
                if score > 60:
                    tag_scores.append((tag, score))
        
        # Sort by score descending and remove duplicates
        seen_tags = set()
        unique_tags = []
        for tag, score in sorted(tag_scores, key=lambda x: x[1], reverse=True):
            if tag not in seen_tags:
                seen_tags.add(tag)
                unique_tags.append(tag)
        
        return unique_tags[:max_results]