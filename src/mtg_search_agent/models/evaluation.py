from typing import List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
from .card import Card


class BaseCardScore(BaseModel):
    """Base class for card scores with common fields"""
    score: int = Field(ge=1, le=10, description="Relevance score from 1-10")
    reasoning: Optional[str] = None


class CardScore(BaseCardScore):
    """Represents a scored card with relevance rating"""
    card: Card


class LightweightCardScore(BaseCardScore):
    """Lightweight card score with only essential data for evaluation"""
    card_id: str = Field(min_length=1, description="Scryfall card ID")
    name: str = Field(min_length=1, description="Card name")


# Type variable for generic evaluation results
T = TypeVar('T', CardScore, LightweightCardScore)


class BaseEvaluationResult(BaseModel, Generic[T]):
    """Base class for evaluation results with common fields"""
    scored_cards: List[T]
    average_score: float
    should_continue: bool
    feedback_for_query_agent: Optional[str] = None
    iteration_count: int


class EvaluationResult(BaseEvaluationResult[CardScore]):
    """Represents the evaluation of search results with full card data"""
    pass


class LightweightEvaluationResult(BaseEvaluationResult[LightweightCardScore]):
    """Complete evaluation result with lightweight card data"""
    pass


# Keep LightweightAgentResult for backward compatibility, but it's now just an alias
# This can be removed in a future version after updating all references
class LightweightAgentResult(BaseModel):
    """Result from evaluation agent - only what the agent should generate
    
    DEPRECATED: Use LightweightEvaluationResult instead. This class is kept for backward compatibility.
    """
    scored_cards: List[LightweightCardScore]
    feedback_for_query_agent: Optional[str] = None