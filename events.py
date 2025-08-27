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
    SCRYFALL_SEARCH_STARTED = "scryfall_search_started"
    CARDS_FOUND = "cards_found"
    CACHE_ANALYZED = "cache_analyzed"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_BATCH_PROGRESS = "evaluation_batch_progress"
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