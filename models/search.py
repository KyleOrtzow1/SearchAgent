from typing import List, Optional
from pydantic import BaseModel
from .card import Card


class SearchQuery(BaseModel):
    """Represents a Scryfall search query"""
    query: str
    explanation: Optional[str] = None
    confidence: Optional[float] = None


class SearchResult(BaseModel):
    """Represents the result of a card search"""
    query: SearchQuery
    cards: List[Card]
    total_cards: int
    has_more: bool = False
    
    
class TagSuggestion(BaseModel):
    """Represents a suggested tag from fuzzy matching"""
    tag: str
    score: float
    category: str