from typing import List, Optional
from pydantic import BaseModel, Field
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
    total_cards: int = Field(ge=0, description="Total number of cards found (must be >= 0)")
    has_more: bool = False
    
    
class TagSuggestion(BaseModel):
    """Represents a suggested tag from fuzzy matching"""
    tag: str
    score: float
    category: str