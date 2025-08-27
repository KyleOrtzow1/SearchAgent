"""
Event system for MTG Search Agent
Provides clean separation between business logic and presentation
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime


class SearchEventType(str, Enum):
    """Types of search events that can be emitted"""
    SEARCH_STARTED = "search_started"
    ITERATION_STARTED = "iteration_started"
    QUERY_GENERATED = "query_generated"
    QUERY_GENERATION_STARTED = "query_generation_started"
    QUERY_STREAMING_PROGRESS = "query_streaming_progress"
    SCRYFALL_SEARCH_STARTED = "scryfall_search_started"
    SCRYFALL_PAGINATION_STARTED = "scryfall_pagination_started"
    SCRYFALL_PAGE_FETCHED = "scryfall_page_fetched"
    SCRYFALL_PAGINATION_COMPLETED = "scryfall_pagination_completed"
    CARDS_FOUND = "cards_found"
    CACHE_ANALYZED = "cache_analyzed"
    EVALUATION_STRATEGY_SELECTED = "evaluation_strategy_selected"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_STREAMING_PROGRESS = "evaluation_streaming_progress"
    EVALUATION_BATCH_PROGRESS = "evaluation_batch_progress"
    EVALUATION_PARALLEL_METRICS = "evaluation_parallel_metrics"
    EVALUATION_COMPLETED = "evaluation_completed"
    ITERATION_COMPLETED = "iteration_completed"
    SEARCH_COMPLETED = "search_completed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class SearchEvent:
    """Base search event data structure"""
    event_type: SearchEventType
    timestamp: datetime
    data: Dict[str, Any]


class SearchEventEmitter:
    """Event emitter for search progress and status updates"""
    
    def __init__(self):
        self._listeners: Dict[SearchEventType, List[Callable]] = {}
        
    def on(self, event_type: SearchEventType, callback: Callable[[Dict[str, Any]], None]):
        """Register an event listener"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def off(self, event_type: SearchEventType, callback: Callable):
        """Remove an event listener"""
        if event_type in self._listeners:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb != callback
            ]
    
    def emit(self, event_type: SearchEventType, data: Dict[str, Any] = None):
        """Emit an event to all registered listeners"""
        if data is None:
            data = {}
            
        event = SearchEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        
        # Call all registered listeners for this event type
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(event.data)
                except Exception as e:
                    # Don't let listener errors break the search
                    print(f"Event listener error: {e}")
    
    async def emit_async(self, event_type: SearchEventType, data: Dict[str, Any] = None):
        """Emit an event asynchronously"""
        if data is None:
            data = {}
            
        event = SearchEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        
        # Call all registered listeners for this event type
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event.data)
                    else:
                        callback(event.data)
                except Exception as e:
                    print(f"Event listener error: {e}")
    
    def clear_listeners(self, event_type: Optional[SearchEventType] = None):
        """Clear all listeners for a specific event type, or all listeners"""
        if event_type:
            self._listeners[event_type] = []
        else:
            self._listeners.clear()


# Predefined event data structures for type safety and documentation

def create_search_started_event(query: str, max_iterations: int) -> Dict[str, Any]:
    """Create data for search started event"""
    return {
        'query': query,
        'max_iterations': max_iterations,
        'timestamp': datetime.now().isoformat()
    }

def create_iteration_started_event(iteration: int, max_iterations: int, elapsed_time: str) -> Dict[str, Any]:
    """Create data for iteration started event"""
    return {
        'iteration': iteration,
        'max_iterations': max_iterations,
        'elapsed_time': elapsed_time
    }

def create_query_generated_event(scryfall_query: str, explanation: str, iteration: int) -> Dict[str, Any]:
    """Create data for query generated event"""
    return {
        'scryfall_query': scryfall_query,
        'explanation': explanation,
        'iteration': iteration
    }

def create_cards_found_event(count: int, total_available: int, paginated: bool, iteration: int) -> Dict[str, Any]:
    """Create data for cards found event"""
    return {
        'count': count,
        'total_available': total_available,
        'paginated': paginated,
        'iteration': iteration
    }

def create_cache_analyzed_event(new_cards: int, cached_cards: int) -> Dict[str, Any]:
    """Create data for cache analyzed event"""
    return {
        'new_cards': new_cards,
        'cached_cards': cached_cards
    }

def create_evaluation_started_event(card_count: int, batch_size: int, parallel: bool) -> Dict[str, Any]:
    """Create data for evaluation started event"""
    return {
        'card_count': card_count,
        'batch_size': batch_size,
        'parallel': parallel
    }

def create_evaluation_progress_event(completed_batches: int, total_batches: int, cards_evaluated: int) -> Dict[str, Any]:
    """Create data for evaluation progress event"""
    return {
        'completed_batches': completed_batches,
        'total_batches': total_batches,
        'cards_evaluated': cards_evaluated,
        'progress_percent': (completed_batches / total_batches) * 100 if total_batches > 0 else 0
    }

def create_evaluation_completed_event(total_cards: int, average_score: float, duration: str) -> Dict[str, Any]:
    """Create data for evaluation completed event"""
    return {
        'total_cards': total_cards,
        'average_score': average_score,
        'duration': duration
    }

def create_iteration_completed_event(iteration: int, score: float, is_best: bool, satisfied: bool, duration: str) -> Dict[str, Any]:
    """Create data for iteration completed event"""
    return {
        'iteration': iteration,
        'score': score,
        'is_best': is_best,
        'satisfied': satisfied,
        'duration': duration
    }

def create_search_completed_event(total_cards: int, final_score: float, iterations: int, duration: str) -> Dict[str, Any]:
    """Create data for search completed event"""
    return {
        'total_cards': total_cards,
        'final_score': final_score,
        'iterations_used': iterations,
        'total_duration': duration
    }

def create_query_generation_started_event(request: str, iteration: int) -> Dict[str, Any]:
    """Create data for query generation started event"""
    return {
        'request': request,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat()
    }

def create_query_streaming_progress_event(partial_query: str, partial_explanation: str) -> Dict[str, Any]:
    """Create data for query streaming progress event"""
    return {
        'partial_query': partial_query,
        'partial_explanation': partial_explanation
    }

def create_scryfall_pagination_started_event(query: str, estimated_total: int = None) -> Dict[str, Any]:
    """Create data for scryfall pagination started event"""
    return {
        'query': query,
        'estimated_total': estimated_total
    }

def create_scryfall_page_fetched_event(page: int, cards_in_page: int, total_cards_so_far: int, total_available: int) -> Dict[str, Any]:
    """Create data for scryfall page fetched event"""
    return {
        'page': page,
        'cards_in_page': cards_in_page,
        'total_cards_so_far': total_cards_so_far,
        'total_available': total_available,
        'progress_percent': (total_cards_so_far / total_available) * 100 if total_available > 0 else 0
    }

def create_scryfall_pagination_completed_event(total_cards: int, pages_fetched: int, limited_by_max: bool) -> Dict[str, Any]:
    """Create data for scryfall pagination completed event"""
    return {
        'total_cards': total_cards,
        'pages_fetched': pages_fetched,
        'limited_by_max': limited_by_max
    }

def create_evaluation_streaming_progress_event(cards_evaluated: int, total_cards: int, current_score: float = None, batch_info: tuple = None) -> Dict[str, Any]:
    """Create data for evaluation streaming progress event"""
    data = {
        'cards_evaluated': cards_evaluated,
        'total_cards': total_cards,
        'progress_percent': (cards_evaluated / total_cards) * 100 if total_cards > 0 else 0
    }
    if current_score is not None:
        data['current_score'] = current_score
    if batch_info:
        batch_index, total_batches = batch_info
        data['batch_index'] = batch_index
        data['total_batches'] = total_batches
    return data

def create_error_event(error_type: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create data for error event"""
    data = {
        'error_type': error_type,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    if context:
        data['context'] = context
    return data

def create_evaluation_strategy_selected_event(strategy: str, card_count: int, batch_size: int = None, total_batches: int = None, reason: str = None) -> Dict[str, Any]:
    """Create data for evaluation strategy selected event"""
    data = {
        'strategy': strategy,
        'card_count': card_count
    }
    if batch_size is not None:
        data['batch_size'] = batch_size
    if total_batches is not None:
        data['total_batches'] = total_batches
    if reason:
        data['reason'] = reason
    return data

def create_parallel_evaluation_metrics_event(total_batches: int, elapsed_time: float, time_saved: float = None, estimated_sequential: float = None) -> Dict[str, Any]:
    """Create data for parallel evaluation performance metrics event"""
    data = {
        'total_batches': total_batches,
        'elapsed_time': elapsed_time
    }
    if time_saved is not None:
        data['time_saved'] = time_saved
    if estimated_sequential is not None:
        data['estimated_sequential'] = estimated_sequential
    return data