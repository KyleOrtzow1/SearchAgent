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
    SearchEventEmitter,
    # New streamlined event classes
    SearchStartedEvent, SearchCompletedEvent, FinalResultsDisplayEvent,
    IterationStartedEvent, QueryGeneratedEvent, ScryfallSearchStartedEvent,
    CardsFoundEvent, CacheAnalyzedEvent, EvaluationStartedEvent,
    EvaluationCompletedEvent, IterationCompletedEvent,
    EvaluationStrategySelectedEvent
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
        return f"{minutes}m {seconds:.0f}s"


class SearchOrchestrator:
    """Main orchestrator that coordinates the search agents"""
    
    def __init__(self):
        self.events = SearchEventEmitter()
        self.query_agent = QueryAgent(event_emitter=self.events)
        self.evaluation_agent = EvaluationAgent(event_emitter=self.events)
        self.scryfall_api = ScryfallAPI(event_emitter=self.events)
        self.max_loops = MAX_SEARCH_LOOPS
        self.card_cache: Dict[str, CardScore] = {}
        
    async def run(self, natural_language_request: str) -> EvaluationResult:
        """
        Execute the search process
        """
        start_time = time.time()
        
        # Initialize session state
        session_state = {
            "previous_queries": [],
            "feedback": None,
            "best_result": None,
            "history": []
        }
        
        # Emit search started event
        self.events.emit(SearchStartedEvent(natural_language_request, self.max_loops))
        
        final_result = None
        
        for i in range(self.max_loops):
            iteration = i + 1
            iteration_start = time.time()
            
            # Emit iteration started event
            self.events.emit(IterationStartedEvent(
                iteration, 
                self.max_loops, 
                format_time_elapsed(start_time)
            ))
            
            # 1. Generate Query
            query = await self._generate_query_step(natural_language_request, session_state, iteration)
            session_state["previous_queries"].append(query.query)
            
            # 2. Search Scryfall
            # ScryfallAPI is synchronous and takes a SearchQuery object
            search_result = self.scryfall_api.search_cards(query)
            cards = search_result.cards
            
            # 3. Evaluate Cards
            if cards:
                evaluation = await self._evaluate_cards_step(
                    natural_language_request, 
                    cards, 
                    iteration, 
                    session_state["previous_queries"],
                    len(cards)
                )
                
                # Aggregate results with cache
                final_result = self._aggregate_iteration_results(
                    list(self.card_cache.values()),
                    evaluation,
                    iteration
                )
            else:
                # No cards found, create empty evaluation
                final_result = self._create_empty_evaluation(iteration)
            
            # 4. Summarize and Decide
            should_stop = self._summarize_iteration_results(
                final_result, 
                session_state, 
                iteration_start
            )
            
            if should_stop:
                break
        
        # Create final result with top cards from cache
        final_result = self._create_final_result_with_top_cards(final_result)
        
        # Emit search completed event
        duration = format_time_elapsed(start_time)
        self.events.emit(SearchCompletedEvent(
            len(final_result.scored_cards),
            final_result.average_score,
            iteration,
            duration
        ))
        
        # Display final results
        self.print_final_results(final_result)
        
        return final_result

    async def _generate_query_step(self, natural_language_request: str, session_state: Dict, iteration: int) -> SearchQuery:
        """Execute query generation step"""
        query = await self.query_agent.generate_query(
            natural_language_request=natural_language_request,
            previous_queries=session_state["previous_queries"],
            feedback=session_state["feedback"]
        )
        
        # Emit query generated event
        self.events.emit(QueryGeneratedEvent(
            query.query,
            query.explanation,
            iteration
        ))
        
        return query

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
        self.events.emit(EvaluationStartedEvent(len(cards), batch_size, ENABLE_PARALLEL_EVALUATION))

        # Evaluate cards
        lightweight_evaluation = await self.evaluation_agent.evaluate_cards(
            natural_language_request=natural_language_request,
            cards=cards,
            iteration_count=iteration,
            previous_queries=previous_queries,
            total_cards=total_cards
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
        # Update cache with new scores
        for scored_card in new_evaluation.scored_cards:
            self.card_cache[scored_card.card.id] = scored_card
            
        # Combine cached and new evaluations (re-fetching from cache to get everything)
        all_scored_cards = list(self.card_cache.values())
        
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
        self.events.emit(IterationCompletedEvent(
            evaluation.iteration_count,
            evaluation.average_score,
            is_new_best,
            is_satisfied,
            duration
        ))
        
        # Return whether to stop the search
        return is_satisfied
    
    def _create_final_result_with_top_cards(self, result: Optional[EvaluationResult]) -> EvaluationResult:
        """Create final result with top N highest scoring cards from entire search cache"""
        # Get all cached cards and sort by score descending
        all_cached_cards = list(self.card_cache.values())
        top_cards = sorted(all_cached_cards, key=lambda x: x.score, reverse=True)[:TOP_CARDS_TO_DISPLAY]
        
        # Calculate stats for the top cards
        if top_cards:
            top_scores = [card.score for card in top_cards]
            top_average = sum(top_scores) / len(top_scores)
        elif result:
            top_average = result.average_score
            top_cards = result.scored_cards[:TOP_CARDS_TO_DISPLAY]
        else:
            # No results found at all
            top_average = 0.0
            top_cards = []

        # Create new result with top cards
        return EvaluationResult(
            scored_cards=top_cards,
            average_score=top_average,
            should_continue=result.should_continue if result else False,
            feedback_for_query_agent=result.feedback_for_query_agent if result else "No cards found matching the search criteria.",
            iteration_count=result.iteration_count if result else 1
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
        
        # Emit the final results display event using new streamlined API
        self.events.emit(FinalResultsDisplayEvent(result, cache_stats))