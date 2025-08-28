import asyncio
import time
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()
from .agents.query_agent import QueryAgent
from .agents.evaluation_agent import EvaluationAgent
from .tools.scryfall_api import ScryfallAPI
from .models.evaluation import EvaluationResult, CardScore, LightweightEvaluationResult
from .models.search import SearchQuery
from .models.card import Card
from .config import MAX_SEARCH_LOOPS, ENABLE_PARALLEL_EVALUATION, EVALUATION_BATCH_SIZE, TOP_CARDS_TO_DISPLAY, ENABLE_FULL_PAGINATION
from .events import (
    SearchEventEmitter, SearchEventType,
    create_search_started_event, create_iteration_started_event, create_query_generated_event,
    create_cards_found_event, create_cache_analyzed_event, create_evaluation_started_event,
    create_evaluation_progress_event, create_evaluation_completed_event,
    create_iteration_completed_event, create_search_completed_event, create_final_results_display_event
)


def format_time_elapsed(start_time: float) -> str:
    """Format elapsed time in a human-readable way"""
    elapsed = time.time() - start_time
    if elapsed < 1:
        return f"{elapsed*1000:.0f}ms"
    elif elapsed < 60:
        return f"{elapsed:.1f}s"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"


class SearchOrchestrator:
    """Main orchestrator that coordinates the search agents"""
    
    def __init__(self, enable_streaming: bool = False):
        self.events = SearchEventEmitter()
        self.query_agent = QueryAgent(event_emitter=self.events)
        self.evaluation_agent = EvaluationAgent(event_emitter=self.events)
        self.scryfall_api = ScryfallAPI(event_emitter=self.events)
        self.max_loops = MAX_SEARCH_LOOPS
        self.card_cache: Dict[str, CardScore] = {}
        self.enable_streaming = enable_streaming
    
    async def search(self, natural_language_request: str) -> EvaluationResult:
        """
        Main search loop that coordinates between query and evaluation agents
        
        Args:
            natural_language_request: User's natural language request for cards
            
        Returns:
            Final evaluation result with best cards found
        """
        start_time = time.time()
        session_state = self._initialize_search_session(natural_language_request)
        
        # Emit search started event
        self.events.emit(
            SearchEventType.SEARCH_STARTED,
            create_search_started_event(natural_language_request, self.max_loops)
        )
        
        # Execute main search loop
        evaluation = None
        for iteration in range(1, self.max_loops + 1):
            iteration_start = time.time()
            
            # Emit iteration started event
            self.events.emit(
                SearchEventType.ITERATION_STARTED,
                create_iteration_started_event(iteration, self.max_loops, format_time_elapsed(start_time))
            )
            
            # Execute search iteration steps
            query = await self._generate_query_step(natural_language_request, session_state, iteration)
            search_result = self._execute_scryfall_search_step(query, iteration)
            
            if not search_result.cards:
                session_state["feedback"] = "No cards found with this query. Try different search terms or broader criteria."
                continue
            
            new_cards, cached_scores = self._analyze_cache_step(search_result.cards)
            
            if new_cards:
                new_evaluation = await self._evaluate_cards_step(
                    natural_language_request, new_cards, iteration, 
                    session_state["previous_queries"], len(search_result.cards)
                )
                # Cache the new evaluations
                for scored_card in new_evaluation.scored_cards:
                    self.card_cache[scored_card.card.id] = scored_card
            else:
                new_evaluation = self._create_empty_evaluation(iteration)
            
            # Aggregate results and check continuation
            evaluation = self._aggregate_iteration_results(cached_scores, new_evaluation, iteration)
            should_stop = self._summarize_iteration_results(evaluation, session_state, iteration_start)
            
            if should_stop:
                break
            
            if iteration == self.max_loops:
                break
        
        total_time = format_time_elapsed(start_time)
        
        # Create final result with top N highest scoring cards from entire search
        final_result = self._create_final_result_with_top_cards(session_state["best_result"] or evaluation)
        
        # Emit search completed event
        self.events.emit(
            SearchEventType.SEARCH_COMPLETED,
            create_search_completed_event(
                len(final_result.scored_cards),
                final_result.average_score,
                final_result.iteration_count,
                total_time
            )
        )
        
        return final_result
    
    def _initialize_search_session(self, natural_language_request: str) -> Dict:
        """Initialize search session state"""
        # Clear cache at start of new search session
        self.card_cache.clear()
        
        return {
            "previous_queries": [],
            "feedback": None,
            "best_result": None
        }
    
    async def _generate_query_step(self, natural_language_request: str, session_state: Dict, iteration: int) -> SearchQuery:
        """Execute query generation step"""
        
        query = await self.query_agent.generate_query(
            natural_language_request=natural_language_request,
            previous_queries=session_state["previous_queries"],
            feedback=session_state["feedback"],
            use_streaming=self.enable_streaming
        )
        
        # Emit query generated event
        self.events.emit(
            SearchEventType.QUERY_GENERATED,
            create_query_generated_event(query.query, query.explanation or "", iteration)
        )
        
        # Store the query
        session_state["previous_queries"].append(query.query)
        return query
    
    def _execute_scryfall_search_step(self, query: SearchQuery, iteration: int):
        """Execute Scryfall search step"""
        # Emit scryfall search started event
        self.events.emit(SearchEventType.SCRYFALL_SEARCH_STARTED, {'query': query.query})
        
        search_result = self.scryfall_api.search_cards(query)
        
        # Emit cards found event
        if search_result.cards:
            paginated = search_result.total_cards > len(search_result.cards) if search_result.total_cards else False
            self.events.emit(
                SearchEventType.CARDS_FOUND,
                create_cards_found_event(
                    len(search_result.cards),
                    search_result.total_cards or len(search_result.cards),
                    paginated,
                    iteration
                )
            )
        
        return search_result
    
    def _analyze_cache_step(self, cards: List[Card]) -> Tuple[List[Card], List[CardScore]]:
        """Analyze cache and separate new cards from cached cards"""
        new_cards = []
        cached_scores = []
        
        for card in cards:
            if card.id in self.card_cache:
                cached_scores.append(self.card_cache[card.id])
            else:
                new_cards.append(card)
        
        # Emit cache analyzed event
        self.events.emit(
            SearchEventType.CACHE_ANALYZED,
            create_cache_analyzed_event(len(new_cards), len(cached_scores))
        )
        
        return new_cards, cached_scores
    
    async def _evaluate_cards_step(
        self, 
        natural_language_request: str, 
        cards: List[Card], 
        iteration: int, 
        previous_queries: List[str],
        total_cards: int
    ) -> EvaluationResult:
        """Execute card evaluation step"""
        
        # Emit evaluation started event
        batch_size = EVALUATION_BATCH_SIZE if ENABLE_PARALLEL_EVALUATION else len(cards)
        self.events.emit(
            SearchEventType.EVALUATION_STARTED,
            create_evaluation_started_event(len(cards), batch_size, ENABLE_PARALLEL_EVALUATION)
        )
        
        # Create progress callback for evaluation
        def progress_callback(completed: int, total: int):
            self.events.emit(
                SearchEventType.EVALUATION_BATCH_PROGRESS,
                create_evaluation_progress_event(completed, total, completed * batch_size)
            )
        
        # Evaluate cards
        start_eval_time = time.time()
        
        if ENABLE_PARALLEL_EVALUATION:
            lightweight_evaluation = await self.evaluation_agent.evaluate_cards_parallel(
                natural_language_request, cards, iteration,
                previous_queries, self.enable_streaming, total_cards
            )
        else:
            lightweight_evaluation = await self.evaluation_agent.evaluate_cards_bulk(
                natural_language_request, cards, iteration,
                previous_queries, self.enable_streaming, total_cards
            )
        
        # Emit evaluation completed event
        eval_duration = format_time_elapsed(start_eval_time)
        self.events.emit(
            SearchEventType.EVALUATION_COMPLETED,
            create_evaluation_completed_event(len(cards), lightweight_evaluation.average_score, eval_duration)
        )
        
        # Convert lightweight result to full result
        return self._convert_lightweight_to_full_result(lightweight_evaluation, cards)
    
    def _create_empty_evaluation(self, iteration: int) -> EvaluationResult:
        """Create empty evaluation result when using cache only"""
        return EvaluationResult(
            scored_cards=[],
            average_score=0.0,
            should_continue=True,
            iteration_count=iteration
        )
    
    def _aggregate_iteration_results(
        self, 
        cached_scores: List[CardScore], 
        new_evaluation: EvaluationResult, 
        iteration: int
    ) -> EvaluationResult:
        """Aggregate cached and new evaluation results"""
        # Combine cached and new evaluations
        all_scored_cards = cached_scores + new_evaluation.scored_cards
        
        if all_scored_cards:
            combined_average = sum(sc.score for sc in all_scored_cards) / len(all_scored_cards)
        else:
            combined_average = 0.0
        
        # Create combined evaluation result
        return EvaluationResult(
            scored_cards=all_scored_cards,
            average_score=combined_average,
            should_continue=new_evaluation.should_continue if new_evaluation.scored_cards else True,
            feedback_for_query_agent=new_evaluation.feedback_for_query_agent if new_evaluation.scored_cards else "Try different search terms to find new cards.",
            iteration_count=iteration
        )
    
    def _summarize_iteration_results(
        self, 
        evaluation: EvaluationResult, 
        session_state: Dict, 
        iteration_start: float
    ) -> bool:
        """Summarize iteration results and determine if search should continue"""
        
        # Update best result if this is better
        is_new_best = False
        if (not session_state["best_result"] or 
            evaluation.average_score > session_state["best_result"].average_score):
            session_state["best_result"] = evaluation
            is_new_best = True
        
        # Check if evaluation agent is satisfied
        is_satisfied = not evaluation.should_continue
        
        # Update feedback for next iteration
        if evaluation.feedback_for_query_agent:
            session_state["feedback"] = evaluation.feedback_for_query_agent
        
        duration = format_time_elapsed(iteration_start)
        
        # Emit iteration completed event
        self.events.emit(
            SearchEventType.ITERATION_COMPLETED,
            create_iteration_completed_event(
                evaluation.iteration_count,
                evaluation.average_score,
                is_new_best,
                is_satisfied,
                duration
            )
        )
        
        # Return whether to stop the search
        return is_satisfied
    
    def _create_final_result_with_top_cards(self, result: EvaluationResult) -> EvaluationResult:
        """Create final result with top N highest scoring cards from entire search cache"""
        # Get all cached cards and sort by score descending
        all_cached_cards = list(self.card_cache.values())
        top_cards = sorted(all_cached_cards, key=lambda x: x.score, reverse=True)[:TOP_CARDS_TO_DISPLAY]
        
        # Calculate stats for the top cards
        if top_cards:
            top_scores = [card.score for card in top_cards]
            top_average = sum(top_scores) / len(top_scores)
        else:
            top_average = result.average_score
            top_cards = result.scored_cards[:TOP_CARDS_TO_DISPLAY]
        
        # Create new result with top cards
        return EvaluationResult(
            scored_cards=top_cards,
            average_score=top_average,
            should_continue=result.should_continue,
            feedback_for_query_agent=result.feedback_for_query_agent,
            iteration_count=result.iteration_count
        )
    
    def _convert_lightweight_to_full_result(self, lightweight_result: LightweightEvaluationResult, cards: List[Card]) -> EvaluationResult:
        """Convert lightweight evaluation result to full result with complete card data"""
        # Create a mapping of card IDs to cards for quick lookup
        card_map = {card.id: card for card in cards}
        
        # Convert lightweight scores to full scores
        full_scored_cards = []
        for lightweight_score in lightweight_result.scored_cards:
            if lightweight_score.card_id in card_map:
                full_card = card_map[lightweight_score.card_id]
                full_score = CardScore(
                    card=full_card,
                    score=lightweight_score.score,
                    reasoning=lightweight_score.reasoning
                )
                full_scored_cards.append(full_score)
        
        return EvaluationResult(
            scored_cards=full_scored_cards,
            average_score=lightweight_result.average_score,
            should_continue=lightweight_result.should_continue,
            feedback_for_query_agent=lightweight_result.feedback_for_query_agent,
            iteration_count=lightweight_result.iteration_count
        )
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the current card cache"""
        return {
            "cached_cards": len(self.card_cache),
            "cache_size_bytes": sum(len(str(card_score)) for card_score in self.card_cache.values())
        }

    def print_final_results(self, result: EvaluationResult) -> None:
        """Display final search results using event handling system"""
        # Get cache stats for the event
        cache_stats = self.get_cache_stats()
        
        # Emit the final results display event
        self.events.emit(
            SearchEventType.FINAL_RESULTS_DISPLAY,
            create_final_results_display_event(result, cache_stats)
        )