"""
Unit tests for the Events system.

Tests cover:
- BaseEvent abstract class and implementations
- All specific event classes and their data
- SearchEventEmitter functionality
- Event serialization and data handling
- Async event emission
- Event listener management
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.mtg_search_agent.events import (
    BaseEvent, SearchEventEmitter,
    # Search Events
    SearchStartedEvent, SearchCompletedEvent,
    # Iteration Events
    IterationStartedEvent, IterationCompletedEvent,
    # Query Events
    QueryGenerationStartedEvent, QueryStreamingProgressEvent, QueryGeneratedEvent,
    # Scryfall Events
    ScryfallSearchStartedEvent, ScryfallPaginationStartedEvent,
    ScryfallPageFetchedEvent, ScryfallPaginationCompletedEvent,
    CardsFoundEvent, ScryfallCardsFetchedEvent,
    # Evaluation Events
    CacheAnalyzedEvent, EvaluationStrategySelectedEvent,
    EvaluationStartedEvent, EvaluationStreamingProgressEvent,
    EvaluationBatchProgressEvent, EvaluationParallelMetricsEvent,
    EvaluationCompletedEvent,
    # Final Results Events
    FinalResultsDisplayEvent,
    # Error Events
    ErrorOccurredEvent
)


class TestBaseEvent:
    """Test suite for the BaseEvent abstract base class."""

    def test_base_event_is_abstract(self):
        """Test that BaseEvent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvent()

    def test_base_event_timestamp(self):
        """Test that BaseEvent sets timestamp on creation."""
        class TestEvent(BaseEvent):
            @property
            def event_type(self) -> str:
                return "test_event"

        event = TestEvent()
        assert hasattr(event, 'timestamp')
        assert isinstance(event.timestamp, datetime)

    def test_base_event_to_dict_method(self):
        """Test BaseEvent to_dict method excludes private attributes."""
        class TestEvent(BaseEvent):
            def __init__(self, data: str, _private: str = "private"):
                super().__init__()
                self.data = data
                self._private = _private

            @property
            def event_type(self) -> str:
                return "test_event"

        event = TestEvent("test_data")
        event_dict = event.to_dict()

        assert "data" in event_dict
        assert "timestamp" not in event_dict  # Excluded by design
        assert "_private" not in event_dict  # Private attributes excluded
        assert event_dict["data"] == "test_data"


class TestSearchEvents:
    """Test suite for search-related events."""

    def test_search_started_event(self):
        """Test SearchStartedEvent creation and properties."""
        event = SearchStartedEvent("red instant damage", 5)

        assert event.event_type == "search_started"
        assert event.query == "red instant damage"
        assert event.max_iterations == 5
        assert hasattr(event, 'timestamp')

    def test_search_completed_event(self):
        """Test SearchCompletedEvent creation and properties."""
        event = SearchCompletedEvent(25, 7.5, 3, "2.5s")

        assert event.event_type == "search_completed"
        assert event.total_cards == 25
        assert event.final_score == 7.5
        assert event.iterations_used == 3
        assert event.total_duration == "2.5s"

    def test_search_events_to_dict(self):
        """Test search events to_dict serialization."""
        start_event = SearchStartedEvent("test query", 3)
        start_dict = start_event.to_dict()

        assert start_dict["query"] == "test query"
        assert start_dict["max_iterations"] == 3

        complete_event = SearchCompletedEvent(10, 8.0, 2, "1.2s")
        complete_dict = complete_event.to_dict()

        assert complete_dict["total_cards"] == 10
        assert complete_dict["final_score"] == 8.0


class TestIterationEvents:
    """Test suite for iteration-related events."""

    def test_iteration_started_event(self):
        """Test IterationStartedEvent creation and properties."""
        event = IterationStartedEvent(2, 5, "500ms")

        assert event.event_type == "iteration_started"
        assert event.iteration == 2
        assert event.max_iterations == 5
        assert event.elapsed_time == "500ms"

    def test_iteration_completed_event(self):
        """Test IterationCompletedEvent creation and properties."""
        event = IterationCompletedEvent(1, 8.5, True, False, "1.2s")

        assert event.event_type == "iteration_completed"
        assert event.iteration == 1
        assert event.score == 8.5
        assert event.is_best is True
        assert event.satisfied is False
        assert event.duration == "1.2s"


class TestQueryEvents:
    """Test suite for query generation events."""

    def test_query_generation_started_event(self):
        """Test QueryGenerationStartedEvent creation and properties."""
        event = QueryGenerationStartedEvent("red burn spells", 1)

        assert event.event_type == "query_generation_started"
        assert event.request == "red burn spells"
        assert event.iteration == 1

    def test_query_streaming_progress_event(self):
        """Test QueryStreamingProgressEvent creation and properties."""
        event = QueryStreamingProgressEvent("c:r t:inst", "Searching for red")

        assert event.event_type == "query_streaming_progress"
        assert event.partial_query == "c:r t:inst"
        assert event.partial_explanation == "Searching for red"

    def test_query_generated_event(self):
        """Test QueryGeneratedEvent creation and properties."""
        event = QueryGeneratedEvent("c:r t:instant", "Red instant spells", 1)

        assert event.event_type == "query_generated"
        assert event.scryfall_query == "c:r t:instant"
        assert event.explanation == "Red instant spells"
        assert event.iteration == 1


class TestScryfallEvents:
    """Test suite for Scryfall API events."""

    def test_scryfall_search_started_event(self):
        """Test ScryfallSearchStartedEvent creation and properties."""
        event = ScryfallSearchStartedEvent("c:r t:instant")

        assert event.event_type == "scryfall_search_started"
        assert event.query == "c:r t:instant"

    def test_scryfall_pagination_started_event(self):
        """Test ScryfallPaginationStartedEvent creation and properties."""
        event = ScryfallPaginationStartedEvent("c:r", 150)

        assert event.event_type == "scryfall_pagination_started"
        assert event.query == "c:r"
        assert event.estimated_total == 150

    def test_scryfall_pagination_started_no_total(self):
        """Test ScryfallPaginationStartedEvent without estimated total."""
        event = ScryfallPaginationStartedEvent("c:r")

        assert event.estimated_total is None

    def test_scryfall_page_fetched_event(self):
        """Test ScryfallPageFetchedEvent creation and progress calculation."""
        event = ScryfallPageFetchedEvent(2, 20, 40, 100)

        assert event.event_type == "scryfall_page_fetched"
        assert event.page == 2
        assert event.cards_in_page == 20
        assert event.total_cards_so_far == 40
        assert event.total_available == 100
        assert event.progress_percent == 40.0

    def test_scryfall_page_fetched_zero_total(self):
        """Test ScryfallPageFetchedEvent with zero total available."""
        event = ScryfallPageFetchedEvent(1, 0, 0, 0)

        assert event.progress_percent == 0

    def test_scryfall_pagination_completed_event(self):
        """Test ScryfallPaginationCompletedEvent creation and properties."""
        event = ScryfallPaginationCompletedEvent(75, 4, True)

        assert event.event_type == "scryfall_pagination_completed"
        assert event.total_cards == 75
        assert event.pages_fetched == 4
        assert event.limited_by_max is True

    def test_cards_found_event(self):
        """Test CardsFoundEvent creation and properties."""
        event = CardsFoundEvent(25, 150, True, 2)

        assert event.event_type == "cards_found"
        assert event.count == 25
        assert event.total_available == 150
        assert event.paginated is True
        assert event.iteration == 2

    def test_scryfall_cards_fetched_event(self, multiple_cards):
        """Test ScryfallCardsFetchedEvent creation and serialization."""
        event = ScryfallCardsFetchedEvent(multiple_cards, 1, 25)

        assert event.event_type == "scryfall_cards_fetched"
        assert event.cards == multiple_cards
        assert event.count == len(multiple_cards)
        assert event.page == 1
        assert event.total_received == 25

        # Test cards_data serialization
        assert len(event.cards_data) == len(multiple_cards)
        assert event.cards_data[0]["name"] == multiple_cards[0].name

    def test_scryfall_cards_fetched_event_serialization_error(self):
        """Test ScryfallCardsFetchedEvent handles objects without expected attributes gracefully."""
        # Create mock cards without expected attributes
        bad_cards = [Mock()]  # Mock without expected attributes

        event = ScryfallCardsFetchedEvent(bad_cards)

        assert event.event_type == "scryfall_cards_fetched"
        assert event.cards == bad_cards
        # The implementation uses getattr with None defaults, so Mock attributes will be Mock objects
        assert len(event.cards_data) == 1
        assert event.cards_data[0]["id"] is not None  # Mock object for id


class TestEvaluationEvents:
    """Test suite for evaluation-related events."""

    def test_cache_analyzed_event(self):
        """Test CacheAnalyzedEvent creation and properties."""
        event = CacheAnalyzedEvent(15, 5)

        assert event.event_type == "cache_analyzed"
        assert event.new_cards == 15
        assert event.cached_cards == 5

    def test_evaluation_strategy_selected_event(self):
        """Test EvaluationStrategySelectedEvent creation and properties."""
        event = EvaluationStrategySelectedEvent("parallel", 20, 5, 4, "Large batch")

        assert event.event_type == "evaluation_strategy_selected"
        assert event.strategy == "parallel"
        assert event.card_count == 20
        assert event.batch_size == 5
        assert event.total_batches == 4
        assert event.reason == "Large batch"

    def test_evaluation_started_event(self):
        """Test EvaluationStartedEvent creation and properties."""
        event = EvaluationStartedEvent(30, 10, True)

        assert event.event_type == "evaluation_started"
        assert event.card_count == 30
        assert event.batch_size == 10
        assert event.parallel is True

    def test_evaluation_streaming_progress_event(self):
        """Test EvaluationStreamingProgressEvent creation and progress calculation."""
        event = EvaluationStreamingProgressEvent(15, 30, 7.5, (2, 5))

        assert event.event_type == "evaluation_streaming_progress"
        assert event.cards_evaluated == 15
        assert event.total_cards == 30
        assert event.progress_percent == 50.0
        assert event.current_score == 7.5
        assert event.batch_index == 2
        assert event.total_batches == 5

    def test_evaluation_streaming_progress_no_batch_info(self):
        """Test EvaluationStreamingProgressEvent without batch info."""
        event = EvaluationStreamingProgressEvent(10, 20)

        assert event.batch_index is None
        assert event.total_batches is None

    def test_evaluation_batch_progress_event(self):
        """Test EvaluationBatchProgressEvent creation and progress calculation."""
        event = EvaluationBatchProgressEvent(3, 5, 30)

        assert event.event_type == "evaluation_batch_progress"
        assert event.completed_batches == 3
        assert event.total_batches == 5
        assert event.cards_evaluated == 30
        assert event.progress_percent == 60.0

    def test_evaluation_parallel_metrics_event(self):
        """Test EvaluationParallelMetricsEvent creation and properties."""
        event = EvaluationParallelMetricsEvent(5, 2.5, 3.5, 6.0)

        assert event.event_type == "evaluation_parallel_metrics"
        assert event.total_batches == 5
        assert event.elapsed_time == 2.5
        assert event.time_saved == 3.5
        assert event.estimated_sequential == 6.0

    def test_evaluation_completed_event(self):
        """Test EvaluationCompletedEvent creation and properties."""
        event = EvaluationCompletedEvent(25, 7.8, "3.2s")

        assert event.event_type == "evaluation_completed"
        assert event.total_cards == 25
        assert event.average_score == 7.8
        assert event.duration == "3.2s"


class TestFinalResultsEvent:
    """Test suite for final results events."""

    def test_final_results_display_event(self, sample_evaluation_result):
        """Test FinalResultsDisplayEvent creation and serialization."""
        cache_stats = {"cached_cards": 50}
        event = FinalResultsDisplayEvent(sample_evaluation_result, cache_stats)

        assert event.event_type == "final_results_display"
        assert len(event.scored_cards) == len(sample_evaluation_result.scored_cards)
        assert event.total_cards == len(sample_evaluation_result.scored_cards)
        assert event.average_score == sample_evaluation_result.average_score
        assert event.iteration_count == sample_evaluation_result.iteration_count
        assert event.has_results is True
        assert event.total_unique_cards_evaluated == 50

        # Test serialized card data
        first_card = event.scored_cards[0]
        assert "name" in first_card
        assert "score" in first_card
        assert "scryfall_uri" in first_card

    def test_final_results_display_event_no_cache_stats(self, sample_evaluation_result):
        """Test FinalResultsDisplayEvent without cache stats."""
        event = FinalResultsDisplayEvent(sample_evaluation_result)

        assert not hasattr(event, 'total_unique_cards_evaluated')

    def test_final_results_display_event_empty_results(self, sample_card):
        """Test FinalResultsDisplayEvent with empty results."""
        from src.mtg_search_agent.models.evaluation import EvaluationResult

        empty_result = EvaluationResult(
            scored_cards=[],
            average_score=0.0,
            should_continue=True,
            iteration_count=1
        )

        event = FinalResultsDisplayEvent(empty_result)

        assert event.has_results is False
        assert event.total_cards == 0
        assert event.scored_cards == []


class TestErrorEvent:
    """Test suite for error events."""

    def test_error_occurred_event(self):
        """Test ErrorOccurredEvent creation and properties."""
        context = {"query": "test", "iteration": 1}
        event = ErrorOccurredEvent("APIError", "Rate limit exceeded", context)

        assert event.event_type == "error_occurred"
        assert event.error_type == "APIError"
        assert event.message == "Rate limit exceeded"
        assert event.context == context

    def test_error_occurred_event_no_context(self):
        """Test ErrorOccurredEvent without context."""
        event = ErrorOccurredEvent("ValidationError", "Invalid input")

        assert event.context is None


class TestSearchEventEmitter:
    """Test suite for the SearchEventEmitter class."""

    def test_event_emitter_creation(self):
        """Test SearchEventEmitter can be created."""
        emitter = SearchEventEmitter()
        assert emitter._listeners == {}

    def test_event_listener_registration(self):
        """Test registering event listeners."""
        emitter = SearchEventEmitter()
        callback = Mock()

        emitter.on("test_event", callback)

        assert "test_event" in emitter._listeners
        assert callback in emitter._listeners["test_event"]

    def test_event_listener_multiple_for_same_type(self):
        """Test registering multiple listeners for same event type."""
        emitter = SearchEventEmitter()
        callback1 = Mock()
        callback2 = Mock()

        emitter.on("test_event", callback1)
        emitter.on("test_event", callback2)

        assert len(emitter._listeners["test_event"]) == 2

    def test_event_emission(self):
        """Test emitting events to listeners."""
        emitter = SearchEventEmitter()
        callback = Mock()

        emitter.on("search_started", callback)

        event = SearchStartedEvent("test query", 5)
        emitter.emit(event)

        callback.assert_called_once_with(event)

    def test_event_emission_no_listeners(self):
        """Test emitting events when no listeners are registered."""
        emitter = SearchEventEmitter()
        event = SearchStartedEvent("test query", 5)

        # Should not raise an exception
        emitter.emit(event)

    def test_event_emission_wrong_type(self):
        """Test emitting events for different types doesn't call wrong listeners."""
        emitter = SearchEventEmitter()
        callback = Mock()

        emitter.on("search_started", callback)

        # Emit different event type
        event = SearchCompletedEvent(10, 8.0, 1, "1s")
        emitter.emit(event)

        callback.assert_not_called()

    def test_event_listener_removal(self):
        """Test removing event listeners."""
        emitter = SearchEventEmitter()
        callback1 = Mock()
        callback2 = Mock()

        emitter.on("test_event", callback1)
        emitter.on("test_event", callback2)

        emitter.off("test_event", callback1)

        assert callback1 not in emitter._listeners["test_event"]
        assert callback2 in emitter._listeners["test_event"]

    def test_event_listener_removal_nonexistent(self):
        """Test removing non-existent event listener."""
        emitter = SearchEventEmitter()
        callback = Mock()

        # Should not raise an exception
        emitter.off("nonexistent_event", callback)

    def test_clear_listeners_specific_type(self):
        """Test clearing listeners for specific event type."""
        emitter = SearchEventEmitter()
        callback1 = Mock()
        callback2 = Mock()

        emitter.on("event1", callback1)
        emitter.on("event2", callback2)

        emitter.clear_listeners("event1")

        assert emitter._listeners["event1"] == []
        assert callback2 in emitter._listeners["event2"]

    def test_clear_listeners_all(self):
        """Test clearing all listeners."""
        emitter = SearchEventEmitter()
        callback1 = Mock()
        callback2 = Mock()

        emitter.on("event1", callback1)
        emitter.on("event2", callback2)

        emitter.clear_listeners()

        assert emitter._listeners == {}

    def test_event_listener_exception_handling(self):
        """Test that listener exceptions don't break emission."""
        emitter = SearchEventEmitter()

        def failing_callback(event):
            raise Exception("Listener failed")

        def working_callback(event):
            working_callback.called = True

        working_callback.called = False

        emitter.on("search_started", failing_callback)
        emitter.on("search_started", working_callback)

        event = SearchStartedEvent("test", 1)

        # Should not raise exception
        emitter.emit(event)

        # Working callback should still be called
        assert working_callback.called is True

    @pytest.mark.asyncio
    async def test_async_event_emission(self):
        """Test asynchronous event emission."""
        emitter = SearchEventEmitter()
        async_callback = AsyncMock()
        sync_callback = Mock()

        emitter.on("search_started", async_callback)
        emitter.on("search_started", sync_callback)

        event = SearchStartedEvent("test", 1)
        await emitter.emit_async(event)

        async_callback.assert_called_once_with(event)
        sync_callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_async_event_emission_exception_handling(self):
        """Test async event emission handles exceptions properly."""
        emitter = SearchEventEmitter()

        async def failing_async_callback(event):
            raise Exception("Async listener failed")

        def working_callback(event):
            working_callback.called = True

        working_callback.called = False

        emitter.on("search_started", failing_async_callback)
        emitter.on("search_started", working_callback)

        event = SearchStartedEvent("test", 1)

        # Should not raise exception
        await emitter.emit_async(event)

        # Working callback should still be called
        assert working_callback.called is True


class TestEventIntegration:
    """Integration tests for the event system."""

    def test_full_search_event_flow(self):
        """Test a complete search event flow."""
        emitter = SearchEventEmitter()
        events_received = []

        def event_collector(event):
            events_received.append(event)

        # Register for all event types
        event_types = [
            "search_started", "iteration_started", "query_generated",
            "cards_found", "evaluation_completed", "iteration_completed",
            "search_completed"
        ]

        for event_type in event_types:
            emitter.on(event_type, event_collector)

        # Simulate a search flow
        emitter.emit(SearchStartedEvent("red burn", 3))
        emitter.emit(IterationStartedEvent(1, 3, "0ms"))
        emitter.emit(QueryGeneratedEvent("c:r", "Red spells", 1))
        emitter.emit(CardsFoundEvent(25, 25, False, 1))
        emitter.emit(EvaluationCompletedEvent(25, 8.5, "2s"))
        emitter.emit(IterationCompletedEvent(1, 8.5, True, True, "3s"))
        emitter.emit(SearchCompletedEvent(25, 8.5, 1, "3s"))

        assert len(events_received) == 7
        assert events_received[0].event_type == "search_started"
        assert events_received[-1].event_type == "search_completed"

    def test_event_data_consistency(self, multiple_cards):
        """Test that event data remains consistent across serialization."""
        event = ScryfallCardsFetchedEvent(multiple_cards, 1, 10)

        # Test that serialized data matches original
        assert event.count == len(multiple_cards)
        assert len(event.cards_data) == len(multiple_cards)

        for i, card_data in enumerate(event.cards_data):
            original_card = multiple_cards[i]
            assert card_data["name"] == original_card.name
            assert card_data["id"] == original_card.id