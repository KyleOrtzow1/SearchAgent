"""
Event system for MTG Search Agent
Provides clean separation between business logic and presentation
"""

from typing import Dict, List, Callable, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import asyncio
from datetime import datetime
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .models.evaluation import EvaluationResult


class BaseEvent(ABC):
    """Base class for all search events"""

    def __init__(self):
        self.timestamp = datetime.now()

    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type identifier"""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format"""
        data = {}
        for key, value in self.__dict__.items():
            if key != 'timestamp' and not key.startswith('_'):
                data[key] = value
        return data


# Search Events - organized by category

class SearchStartedEvent(BaseEvent):
    def __init__(self, query: str, max_iterations: int):
        super().__init__()
        self.query = query
        self.max_iterations = max_iterations
    
    @property
    def event_type(self) -> str:
        return "search_started"


class SearchCompletedEvent(BaseEvent):
    def __init__(self, total_cards: int, final_score: float, iterations: int, duration: str):
        super().__init__()
        self.total_cards = total_cards
        self.final_score = final_score
        self.iterations_used = iterations
        self.total_duration = duration
    
    @property
    def event_type(self) -> str:
        return "search_completed"


# Iteration Events


class IterationStartedEvent(BaseEvent):
    def __init__(self, iteration: int, max_iterations: int, elapsed_time: str):
        super().__init__()
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.elapsed_time = elapsed_time
    
    @property
    def event_type(self) -> str:
        return "iteration_started"



class IterationCompletedEvent(BaseEvent):
    def __init__(self, iteration: int, score: float, is_best: bool, satisfied: bool, duration: str):
        super().__init__()
        self.iteration = iteration
        self.score = score
        self.is_best = is_best
        self.satisfied = satisfied
        self.duration = duration
    
    @property
    def event_type(self) -> str:
        return "iteration_completed"


# Query Events


class QueryGenerationStartedEvent(BaseEvent):
    def __init__(self, request: str, iteration: int):
        super().__init__()
        self.request = request
        self.iteration = iteration
    
    @property
    def event_type(self) -> str:
        return "query_generation_started"



class QueryStreamingProgressEvent(BaseEvent):
    def __init__(self, partial_query: str, partial_explanation: str):
        super().__init__()
        self.partial_query = partial_query
        self.partial_explanation = partial_explanation
    
    @property
    def event_type(self) -> str:
        return "query_streaming_progress"



class QueryGeneratedEvent(BaseEvent):
    def __init__(self, scryfall_query: str, explanation: str, iteration: int):
        super().__init__()
        self.scryfall_query = scryfall_query
        self.explanation = explanation
        self.iteration = iteration
    
    @property
    def event_type(self) -> str:
        return "query_generated"


# Scryfall Events


class ScryfallSearchStartedEvent(BaseEvent):
    def __init__(self, query: str):
        super().__init__()
        self.query = query
    
    @property
    def event_type(self) -> str:
        return "scryfall_search_started"



class ScryfallPaginationStartedEvent(BaseEvent):
    def __init__(self, query: str, estimated_total: Optional[int] = None):
        super().__init__()
        self.query = query
        self.estimated_total = estimated_total
    
    @property
    def event_type(self) -> str:
        return "scryfall_pagination_started"



class ScryfallPageFetchedEvent(BaseEvent):
    def __init__(self, page: int, cards_in_page: int, total_cards_so_far: int, total_available: int):
        super().__init__()
        self.page = page
        self.cards_in_page = cards_in_page
        self.total_cards_so_far = total_cards_so_far
        self.total_available = total_available
        self.progress_percent = (total_cards_so_far / total_available) * 100 if total_available > 0 else 0
    
    @property
    def event_type(self) -> str:
        return "scryfall_page_fetched"



class ScryfallPaginationCompletedEvent(BaseEvent):
    def __init__(self, total_cards: int, pages_fetched: int, limited_by_max: bool):
        super().__init__()
        self.total_cards = total_cards
        self.pages_fetched = pages_fetched
        self.limited_by_max = limited_by_max
    
    @property
    def event_type(self) -> str:
        return "scryfall_pagination_completed"



class CardsFoundEvent(BaseEvent):
    def __init__(self, count: int, total_available: int, paginated: bool, iteration: int):
        super().__init__()
        self.count = count
        self.total_available = total_available
        self.paginated = paginated
        self.iteration = iteration
    
    @property
    def event_type(self) -> str:
        return "cards_found"


# Evaluation Events


class CacheAnalyzedEvent(BaseEvent):
    def __init__(self, new_cards: int, cached_cards: int):
        super().__init__()
        self.new_cards = new_cards
        self.cached_cards = cached_cards
    
    @property
    def event_type(self) -> str:
        return "cache_analyzed"



class EvaluationStrategySelectedEvent(BaseEvent):
    def __init__(self, strategy: str, card_count: int, batch_size: Optional[int] = None, 
                 total_batches: Optional[int] = None, reason: Optional[str] = None):
        super().__init__()
        self.strategy = strategy
        self.card_count = card_count
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.reason = reason
    
    @property
    def event_type(self) -> str:
        return "evaluation_strategy_selected"



class EvaluationStartedEvent(BaseEvent):
    def __init__(self, card_count: int, batch_size: int, parallel: bool):
        super().__init__()
        self.card_count = card_count
        self.batch_size = batch_size
        self.parallel = parallel
    
    @property
    def event_type(self) -> str:
        return "evaluation_started"



class EvaluationStreamingProgressEvent(BaseEvent):
    def __init__(self, cards_evaluated: int, total_cards: int, current_score: Optional[float] = None, 
                 batch_info: Optional[tuple] = None):
        super().__init__()
        self.cards_evaluated = cards_evaluated
        self.total_cards = total_cards
        self.progress_percent = (cards_evaluated / total_cards) * 100 if total_cards > 0 else 0
        self.current_score = current_score
        if batch_info:
            self.batch_index, self.total_batches = batch_info
        else:
            self.batch_index = None
            self.total_batches = None
    
    @property
    def event_type(self) -> str:
        return "evaluation_streaming_progress"



class EvaluationBatchProgressEvent(BaseEvent):
    def __init__(self, completed_batches: int, total_batches: int, cards_evaluated: int):
        super().__init__()
        self.completed_batches = completed_batches
        self.total_batches = total_batches
        self.cards_evaluated = cards_evaluated
        self.progress_percent = (completed_batches / total_batches) * 100 if total_batches > 0 else 0
    
    @property
    def event_type(self) -> str:
        return "evaluation_batch_progress"



class EvaluationParallelMetricsEvent(BaseEvent):
    def __init__(self, total_batches: int, elapsed_time: float, time_saved: Optional[float] = None, 
                 estimated_sequential: Optional[float] = None):
        super().__init__()
        self.total_batches = total_batches
        self.elapsed_time = elapsed_time
        self.time_saved = time_saved
        self.estimated_sequential = estimated_sequential
    
    @property
    def event_type(self) -> str:
        return "evaluation_parallel_metrics"



class EvaluationCompletedEvent(BaseEvent):
    def __init__(self, total_cards: int, average_score: float, duration: str):
        super().__init__()
        self.total_cards = total_cards
        self.average_score = average_score
        self.duration = duration
    
    @property
    def event_type(self) -> str:
        return "evaluation_completed"


# Final Results Events


class FinalResultsDisplayEvent(BaseEvent):
    def __init__(self, result: "EvaluationResult", cache_stats: Optional[Dict[str, int]] = None):
        super().__init__()
        
        # Serialize the scored cards data
        scored_cards_data = []
        for scored_card in result.scored_cards:
            card_data = {
                'name': scored_card.card.name,
                'score': scored_card.score,
                'mana_cost': scored_card.card.mana_cost,
                'type_line': scored_card.card.type_line,
                'scryfall_uri': scored_card.card.scryfall_uri,
                'reasoning': scored_card.reasoning
            }
            
            # Add power/toughness if available
            if (hasattr(scored_card.card, 'power') and hasattr(scored_card.card, 'toughness') 
                and scored_card.card.power and scored_card.card.toughness):
                card_data['power'] = scored_card.card.power
                card_data['toughness'] = scored_card.card.toughness
                
            scored_cards_data.append(card_data)
        
        self.scored_cards = scored_cards_data
        self.total_cards = len(result.scored_cards)
        self.average_score = result.average_score
        self.iteration_count = result.iteration_count
        self.has_results = len(result.scored_cards) > 0
        
        # Add cache stats if provided
        if cache_stats:
            self.total_unique_cards_evaluated = cache_stats.get('cached_cards', 0)
    
    @property
    def event_type(self) -> str:
        return "final_results_display"


# Error Events


class ErrorOccurredEvent(BaseEvent):
    def __init__(self, error_type: str, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.error_type = error_type
        self.message = message
        self.context = context
    
    @property
    def event_type(self) -> str:
        return "error_occurred"


class SearchEventEmitter:
    """Event emitter for search progress and status updates (new event-class-only API)"""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, callback: Callable[[BaseEvent], None]):
        """Register an event listener. Use the event_type string from the event classes."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: str, callback: Callable):
        """Remove an event listener"""
        if event_type in self._listeners:
            self._listeners[event_type] = [cb for cb in self._listeners[event_type] if cb != callback]

    def emit(self, event: BaseEvent):
        """Emit an event instance to all registered listeners"""
        event_type = event.event_type
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Don't let listener errors break the search
                    print(f"Event listener error: {e}")

    async def emit_async(self, event: BaseEvent):
        """Emit an event asynchronously"""
        event_type = event.event_type
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    print(f"Event listener error: {e}")

    def clear_listeners(self, event_type: Optional[str] = None):
        """Clear all listeners for a specific event type, or all listeners"""
        if event_type:
            self._listeners[event_type] = []
        else:
            self._listeners.clear()