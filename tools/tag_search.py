import json
from typing import List, Dict, Any
from fuzzywuzzy import fuzz
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.search import TagSuggestion


class TagSearchTool:
    """Tool for finding similar tags using fuzzy string matching"""
    
    def __init__(self, tags_file_path: str):
        self.tags_file_path = tags_file_path
        self.tags_data: Dict[str, List[str]] = {}
        self.flat_tags: List[tuple] = []  # (tag, category)
        self._load_tags()
    
    def _load_tags(self) -> None:
        """Load tags from JSON file and create flat structure for searching"""
        with open(self.tags_file_path, 'r', encoding='utf-8') as f:
            self.tags_data = json.load(f)
        
        # Create flat list of (tag, category) tuples for efficient searching
        for category, tags in self.tags_data.items():
            for tag in tags:
                self.flat_tags.append((tag, category))
    
    def find_similar_tags(self, guess_tags: List[str], max_results: int = 10) -> List[TagSuggestion]:
        """
        Find tags similar to the provided guesses using fuzzy matching
        
        Args:
            guess_tags: List of tag guesses from the agent
            max_results: Maximum number of suggestions to return
            
        Returns:
            List of TagSuggestion objects sorted by relevance score
        """
        suggestions = []
        
        for guess in guess_tags:
            # Calculate fuzzy match scores for all tags
            for tag, category in self.flat_tags:
                # Use token_sort_ratio for better matching of multi-word terms
                score = fuzz.token_sort_ratio(guess.lower(), tag.lower())
                
                # Only include matches above a threshold
                if score > 60:
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        score=score / 100.0,  # Normalize to 0-1
                        category=category
                    ))
        
        # Sort by score descending and remove duplicates
        seen_tags = set()
        unique_suggestions = []
        for suggestion in sorted(suggestions, key=lambda x: x.score, reverse=True):
            if suggestion.tag not in seen_tags:
                seen_tags.add(suggestion.tag)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:max_results]