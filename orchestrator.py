import asyncio
import time
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()
from agents.query_agent import QueryAgent
from agents.evaluation_agent import EvaluationAgent
from tools.scryfall_api import ScryfallAPI
from models.evaluation import EvaluationResult, CardScore, LightweightEvaluationResult
from models.search import SearchQuery
from models.card import Card
from config import MAX_SEARCH_LOOPS, ENABLE_PARALLEL_EVALUATION, EVALUATION_BATCH_SIZE, TOP_CARDS_TO_DISPLAY, ENABLE_FULL_PAGINATION


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
    
    def __init__(self, tags_file_path: str, enable_streaming: bool = False):
        self.query_agent = QueryAgent(tags_file_path)
        self.evaluation_agent = EvaluationAgent()
        self.scryfall_api = ScryfallAPI()
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
        # Clear cache at start of new search session
        self.card_cache.clear()
        
        # Track timing
        start_time = time.time()
        
        previous_queries = []
        feedback = None
        best_result = None
        
        print("🚀 Starting MTG Card Search Session")
        print(f"📝 Request: {natural_language_request}")
        print(f"🔢 Max iterations: {self.max_loops}")
        if ENABLE_PARALLEL_EVALUATION:
            print(f"⚡ Parallel evaluation enabled (batch size: {EVALUATION_BATCH_SIZE})")
        else:
            print("📊 Using bulk evaluation mode")
        print("-" * 60)
        
        for iteration in range(1, self.max_loops + 1):
            iteration_start = time.time()
            print(f"\n{'='*60}")
            print(f"🔄 ITERATION {iteration}/{self.max_loops} • ⏱️ {format_time_elapsed(start_time)}")
            print(f"{'='*60}")
            
            # Generate query using query agent
            print(f"\n🎯 STEP 1: Query Generation")
            print("-" * 30)
            query = await self.query_agent.generate_query(
                natural_language_request=natural_language_request,
                previous_queries=previous_queries,
                feedback=feedback,
                use_streaming=self.enable_streaming
            )
            
            print(f"Query: {query.query}")
            if query.explanation:
                print(f"Explanation: {query.explanation}")
            
            # Store the query
            previous_queries.append(query.query)
            
            # Search using Scryfall API
            print(f"\n🔍 STEP 2: Scryfall Search")
            print("-" * 30)
            print("📡 Searching Scryfall API...")
            search_result = self.scryfall_api.search_cards(query)
            
            if not search_result.cards:
                print("❌ No cards found. Trying different approach...")
                feedback = "No cards found with this query. Try different search terms or broader criteria."
                continue
            
            if ENABLE_FULL_PAGINATION:
                pagination_info = f" (paginated from {search_result.total_cards} total available)" if search_result.total_cards > len(search_result.cards) else ""
                print(f"✅ Found {len(search_result.cards)} cards{pagination_info}")
            else:
                print(f"✅ Found {len(search_result.cards)} cards")
            
            # Filter out already-evaluated cards and get new cards to evaluate
            print(f"\n⚡ STEP 3: Cache Analysis")
            print("-" * 30)
            new_cards = []
            cached_scores = []
            
            for card in search_result.cards:
                if card.id in self.card_cache:
                    cached_scores.append(self.card_cache[card.id])
                else:
                    new_cards.append(card)
            
            print(f"🆕 New cards to evaluate: {len(new_cards)}")
            print(f"💾 Cached cards: {len(cached_scores)}")
            
            # Evaluate only new cards
            if new_cards:
                print(f"\n🎯 STEP 4: Card Evaluation")
                print("-" * 30)
                lightweight_evaluation = await self.evaluation_agent.evaluate_cards(
                    natural_language_request=natural_language_request,
                    cards=new_cards,
                    iteration_count=iteration,
                    previous_queries=previous_queries,
                    use_streaming=self.enable_streaming,
                    total_cards=len(search_result.cards)
                )
                
                # Convert lightweight result to full result
                new_evaluation = self._convert_lightweight_to_full_result(lightweight_evaluation, new_cards)
                
                # Cache the new evaluations
                for scored_card in new_evaluation.scored_cards:
                    self.card_cache[scored_card.card.id] = scored_card
            else:
                print(f"\n⚡ STEP 4: Using Cache Only")
                print("-" * 30)
                print("🎯 No new cards to evaluate, using cached results only")
                new_evaluation = EvaluationResult(
                    scored_cards=[],
                    average_score=0.0,
                    should_continue=True,
                    iteration_count=iteration
                )
            
            # Combine cached and new evaluations
            all_scored_cards = cached_scores + new_evaluation.scored_cards
            
            if all_scored_cards:
                combined_average = sum(sc.score for sc in all_scored_cards) / len(all_scored_cards)
            else:
                combined_average = 0.0
            
            # Create combined evaluation result
            evaluation = EvaluationResult(
                scored_cards=all_scored_cards,
                average_score=combined_average,
                should_continue=new_evaluation.should_continue if new_cards else True,
                feedback_for_query_agent=new_evaluation.feedback_for_query_agent if new_cards else "Try different search terms to find new cards.",
                iteration_count=iteration
            )
            
            print(f"\n📊 STEP 5: Results Summary")
            print("-" * 30)
            print(f"📈 Total relevance score: {evaluation.average_score:.1f}/10")
            print(f"🎯 Cards evaluated: {len(evaluation.scored_cards)}")
            
            # Store best result so far
            if best_result is None or evaluation.average_score > best_result.average_score:
                best_result = evaluation
                print(f"🌟 New best result! (Score: {evaluation.average_score:.1f}/10)")
            
            # Check if we should continue
            if not evaluation.should_continue:
                print("✅ Evaluation agent is satisfied with results. Stopping search.")
                break
            
            if iteration == self.max_loops:
                print("🔚 Maximum iterations reached. Returning best result found.")
                break
            
            # Use feedback for next iteration
            feedback = evaluation.feedback_for_query_agent
            if feedback:
                print(f"\n💡 Feedback for next iteration:")
                print(f"   {feedback}")
            
            print(f"⏱️ Iteration completed in {format_time_elapsed(iteration_start)}")
        
        total_time = format_time_elapsed(start_time)
        print(f"\n🏁 Search session completed in {total_time}")
        
        # Create final result with top N highest scoring cards from entire search
        final_result = self._create_final_result_with_top_cards(best_result or evaluation)
        return final_result
    
    def _convert_lightweight_to_full_result(
        self, 
        lightweight_result: LightweightEvaluationResult, 
        cards: List[Card]
    ) -> EvaluationResult:
        """Convert lightweight evaluation result back to full result using card data"""
        # Create a lookup dict for cards by name
        card_lookup = {card.name: card for card in cards}
        
        # Convert lightweight scores to full scores
        full_scored_cards = []
        for lightweight_score in lightweight_result.scored_cards:
            card = card_lookup.get(lightweight_score.name)
            if card:
                full_score = CardScore(
                    card=card,
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
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the current card cache"""
        return {
            "cached_cards": len(self.card_cache),
            "cache_size_bytes": sum(len(str(card_score)) for card_score in self.card_cache.values())
        }
    
    def print_final_results(self, result: EvaluationResult) -> None:
        """Print the final search results in a readable format"""
        print("\n" + "="*60)
        print("FINAL SEARCH RESULTS")
        print("="*60)
        
        if not result.scored_cards:
            print("No relevant cards found.")
            return
        
        # Show cache stats
        cache_stats = self.get_cache_stats()
        total_unique_cards = cache_stats['cached_cards']
        
        print(f"Search completed in {result.iteration_count} iteration(s)")
        print(f"Total unique cards evaluated: {total_unique_cards}")
        print(f"Showing top {min(len(result.scored_cards), TOP_CARDS_TO_DISPLAY)} highest scoring cards (Average: {result.average_score:.1f}/10)")
        print()
        
        # Sort by score descending
        sorted_cards = sorted(result.scored_cards, key=lambda x: x.score, reverse=True)
        
        for i, scored_card in enumerate(sorted_cards, 1):
            card = scored_card.card
            print(f"{i}. {card.name} - Score: {scored_card.score}/10 ⭐")
            print(f"   Mana Cost: {card.mana_cost or 'None'}")
            print(f"   Type: {card.type_line}")
            print(f"   Set: {card.set_name} ({card.set_code.upper()})")
            print(f"   Rarity: {card.rarity.title()}")
            
            if card.power and card.toughness:
                print(f"   Power/Toughness: {card.power}/{card.toughness}")
            
            if card.oracle_text:
                # Truncate long text
                text = card.oracle_text
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"   Text: {text}")
            
            if scored_card.reasoning:
                print(f"   Why: {scored_card.reasoning}")
            
            print(f"   Scryfall: {card.scryfall_uri}")
            print()


async def main():
    """Example usage of the search orchestrator"""
    # This would be called from example.py
    pass


if __name__ == "__main__":
    asyncio.run(main())