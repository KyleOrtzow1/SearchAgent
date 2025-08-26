from typing import List, Optional
from pydantic import BaseModel, Field
from .card import Card


class CardScore(BaseModel):
    """Represents a scored card with relevance rating"""
    card: Card
    score: int = Field(ge=1, le=10, description="Relevance score from 1-10")
    reasoning: Optional[str] = None


class LightweightCardScore(BaseModel):
    """Lightweight card score with only essential data for evaluation"""
    card_id: str = Field(description="Scryfall card ID")
    name: str = Field(description="Card name")
    score: int = Field(ge=1, le=10, description="Relevance score from 1-10")
    reasoning: Optional[str] = None


class LightweightAgentResult(BaseModel):
    """Result from evaluation agent - only what the agent should generate"""
    scored_cards: List[LightweightCardScore]
    feedback_for_query_agent: Optional[str] = None

class LightweightEvaluationResult(BaseModel):
    """Complete evaluation result with calculated fields"""
    scored_cards: List[LightweightCardScore]
    average_score: float
    should_continue: bool
    feedback_for_query_agent: Optional[str] = None
    iteration_count: int


class EvaluationResult(BaseModel):
    """Represents the evaluation of search results"""
    scored_cards: List[CardScore]
    average_score: float
    should_continue: bool
    feedback_for_query_agent: Optional[str] = None
    iteration_count: int