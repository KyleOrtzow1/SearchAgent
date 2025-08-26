from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
import sys
import os
import asyncio
import time
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.card import Card
from models.evaluation import EvaluationResult, CardScore, LightweightEvaluationResult, LightweightAgentResult
from config import ENABLE_PARALLEL_EVALUATION, EVALUATION_BATCH_SIZE, TOP_CARDS_TO_DISPLAY


# Create the evaluation agent with lightweight output
lightweight_evaluation_agent = Agent(
    model=OpenAIResponsesModel('gpt-5-mini'),
    output_type=LightweightAgentResult
)


class EvaluationAgent:
    """Agent responsible for evaluating card relevance and providing feedback"""
    
    def _build_evaluation_prompt(
        self,
        natural_language_request: str,
        cards: List[Card],
        iteration_count: int,
        total_cards: int,
        previous_queries: List[str] = None,
        batch_info: tuple = None
    ) -> str:
        """
        Build evaluation prompt for cards
        
        Args:
            natural_language_request: The original user request
            cards: List of cards to evaluate
            iteration_count: Current iteration number
            total_cards: Total number of cards found in the search
            previous_queries: List of queries that have been tried
            batch_info: Optional tuple of (batch_index, total_batches) for batch evaluation
            
        Returns:
            Formatted prompt string
        """
        previous_queries = previous_queries or []
        
        # Build card summaries
        card_summaries = []
        for i, card in enumerate(cards, 1):
            card_info = f"{i}. {card.name} ({card.mana_cost or 'No cost'}) - {card.type_line}"
            if card.oracle_text:
                card_info += f" | {card.oracle_text}"
            card_summaries.append(card_info)
        
        # Start building prompt
        prompt_parts = [f"Original request: {natural_language_request}", ""]
        
        # Add batch info or regular cards header
        if batch_info:
            batch_index, total_batches = batch_info
            prompt_parts.append(f"Cards to evaluate (Batch {batch_index + 1}/{total_batches} - showing {len(cards)} of {total_cards} total cards found):")
        else:
            prompt_parts.append("Cards found:")
        
        prompt_parts.extend(card_summaries)
        
        # Add previous queries if available
        if previous_queries:
            prompt_parts.extend(["", f"Previous queries tried: {', '.join(previous_queries)}"])
        
        # Add evaluation instructions
        prompt_parts.extend([
            "",
            f"This is iteration {iteration_count} of the search process.",
            "",
            "Evaluate each card's relevance to the original request on a scale of 1-10.",
            "Score 1-3: Not relevant, Score 4-6: Somewhat relevant, Score 7-10: Highly relevant",
            "",
            "Return a LightweightAgentResult with:",
            "- scored_cards: List with ONLY name (string), score (integer 1-10), and reasoning (string) for each card",
            f"- feedback_for_query_agent: If scores are low OR fewer than {TOP_CARDS_TO_DISPLAY} total cards found (currently {total_cards} total), provide specific suggestions for broader search terms",
            ""
        ])
        
        # Add final instructions for batch evaluation
        if batch_info:
            prompt_parts.extend([
                "CRITICAL: Every scored_card MUST have exactly these fields:",
                "  - name: string (card name)",
                "  - score: integer between 1 and 10",  
                "  - reasoning: string (why this score)",
                ""
            ])
        
        prompt_parts.append("DO NOT include full card data - just name, score, and reasoning!")
        
        return "\n".join(prompt_parts)
    
    def _calculate_metrics(
        self,
        agent_result: LightweightAgentResult,
        total_cards: int,
        iteration_count: int
    ) -> tuple:
        """
        Calculate evaluation metrics and determine if search should continue
        
        Args:
            agent_result: Result from the evaluation agent
            total_cards: Total number of cards found in the search
            iteration_count: Current iteration number
            
        Returns:
            Tuple of (average_score, should_continue)
        """
        # Calculate average score from individual card scores
        scores = [card.score for card in agent_result.scored_cards]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Check if we need to continue searching based on score OR insufficient total card count
        has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
        should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
        
        return avg_score, should_continue
    
    def _create_evaluation_result(
        self,
        agent_result: LightweightAgentResult,
        avg_score: float,
        should_continue: bool,
        iteration_count: int
    ) -> LightweightEvaluationResult:
        """
        Create a LightweightEvaluationResult from agent output and calculated metrics
        
        Args:
            agent_result: Result from the evaluation agent
            avg_score: Calculated average score
            should_continue: Whether search should continue
            iteration_count: Current iteration number
            
        Returns:
            Complete evaluation result
        """
        return LightweightEvaluationResult(
            scored_cards=agent_result.scored_cards,
            average_score=avg_score,
            should_continue=should_continue,
            feedback_for_query_agent=agent_result.feedback_for_query_agent,
            iteration_count=iteration_count
        )
    
    def _combine_batch_results(
        self,
        batch_results: List[LightweightEvaluationResult],
        total_cards: int,
        iteration_count: int
    ) -> LightweightEvaluationResult:
        """
        Combine results from multiple batch evaluations
        
        Args:
            batch_results: List of results from individual batches
            total_cards: Total number of cards found in the search
            iteration_count: Current iteration number
            
        Returns:
            Combined evaluation result
        """
        # Combine results from all batches
        all_scored_cards = []
        all_scores = []
        
        for batch_result in batch_results:
            all_scored_cards.extend(batch_result.scored_cards)
            # Extract individual scores for overall average
            for scored_card in batch_result.scored_cards:
                all_scores.append(scored_card.score)
        
        # Calculate combined metrics
        combined_average = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Determine if search should continue based on combined average OR insufficient total card count
        has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
        should_continue = (combined_average < 6 or has_insufficient_cards) and iteration_count < 5
        
        # Generate combined feedback from all batches if needed
        feedback = None
        if should_continue:
            # Collect all non-empty feedback from individual batches
            batch_feedbacks = [result.feedback_for_query_agent for result in batch_results 
                             if result.feedback_for_query_agent and result.feedback_for_query_agent.strip()]
            
            low_scores = [score for score in all_scores if score < 6]
            
            if has_insufficient_cards:
                feedback = f"Found only {total_cards} cards but need {TOP_CARDS_TO_DISPLAY}. Try broader search terms or different criteria to find more relevant cards."
            elif len(low_scores) > len(all_scores) * 0.5:  # If more than 50% are low scored
                feedback = f"Average relevance is {combined_average:.1f}/10. "
                if batch_feedbacks:
                    # Aggregate unique suggestions from batches
                    unique_suggestions = set()
                    for batch_feedback in batch_feedbacks:
                        # Extract suggestion keywords from each batch feedback
                        if "broader" in batch_feedback.lower():
                            unique_suggestions.add("broader search terms")
                        if "different" in batch_feedback.lower():
                            unique_suggestions.add("different criteria")
                        if "more specific" in batch_feedback.lower():
                            unique_suggestions.add("more specific terms")
                    
                    if unique_suggestions:
                        feedback += f"Try {', '.join(unique_suggestions)} to find more relevant cards."
                    else:
                        feedback += "Try different search terms or broader criteria to find more relevant cards."
                else:
                    feedback += "Try different search terms or broader criteria to find more relevant cards."
        
        return LightweightEvaluationResult(
            scored_cards=all_scored_cards,
            average_score=combined_average,
            should_continue=should_continue,
            feedback_for_query_agent=feedback,
            iteration_count=iteration_count
        )
    
    async def _execute_agent(
        self,
        prompt: str,
        use_streaming: bool,
        cards_count: int,
        batch_info: tuple = None
    ) -> LightweightAgentResult:
        """
        Execute the evaluation agent with either streaming or direct generation
        
        Args:
            prompt: Evaluation prompt to send to agent
            use_streaming: Whether to use streaming mode
            cards_count: Number of cards being evaluated
            batch_info: Optional tuple of (batch_index, total_batches) for batch evaluation
            
        Returns:
            Agent result with scored cards and feedback
        """
        if use_streaming:
            # Show progress message
            if batch_info:
                batch_index, total_batches = batch_info
                print(f"🧠 Batch {batch_index + 1}/{total_batches} evaluating {cards_count} cards...", end="", flush=True)
            else:
                print("📊 Evaluation Agent analyzing cards...")
                print(f"   Evaluating {cards_count} cards for relevance...")
                print("🧠 ", end="", flush=True)
            
            try:
                async with lightweight_evaluation_agent.run_stream(prompt) as result:
                    # Stream structured output as it's being built
                    cards_evaluated = 0
                    async for partial_evaluation in result.stream():
                        # Show evaluation progress
                        if hasattr(partial_evaluation, 'scored_cards'):
                            current_cards = len(partial_evaluation.scored_cards)
                            if current_cards > cards_evaluated:
                                cards_evaluated = current_cards
                                if batch_info:
                                    batch_index, total_batches = batch_info
                                    print(f"\r🧠 Batch {batch_index + 1}/{total_batches}: {cards_evaluated}/{cards_count} cards", end="", flush=True)
                                else:
                                    print(f"\r🧠 Evaluated {cards_evaluated}/{cards_count} cards", end="", flush=True)
                        
                        # Show average score as it develops
                        if hasattr(partial_evaluation, 'average_score') and partial_evaluation.average_score > 0:
                            score_text = f" | Score: {partial_evaluation.average_score:.1f}" if batch_info else f" | Avg Score: {partial_evaluation.average_score:.1f}"
                            print(score_text, end="", flush=True)
                
                if batch_info:
                    print(" ✅")
                else:
                    print(f"\n📈 Card evaluation complete!")
                
                return await result.get_output()
                
            except Exception as e:
                if batch_info:
                    batch_index, total_batches = batch_info
                    print(f" ⚠️ Error: {e}")
                    print(f"🔄 Batch {batch_index + 1} falling back to non-streaming...")
                else:
                    print(f"\n⚠️ Streaming error: {e}")
                    print("🔄 Falling back to non-streaming mode...")
                
                # Fallback to non-streaming
                result = await lightweight_evaluation_agent.run(prompt)
                return result.output
        else:
            # Use direct generation without streaming
            if batch_info:
                batch_index, total_batches = batch_info
                print(f"🧠 Batch {batch_index + 1}/{total_batches} evaluating {cards_count} cards...", end="", flush=True)
            else:
                print("📊 Evaluation Agent analyzing cards...")
                print(f"   Evaluating {cards_count} cards for relevance...")
            
            try:
                result = await lightweight_evaluation_agent.run(prompt)
                if batch_info:
                    print(" ✅")
                else:
                    print("📈 Card evaluation complete!")
                return result.output
            except Exception as e:
                if batch_info:
                    print(f" ❌ Error: {e}")
                    # Return empty result for this batch to prevent total failure
                    return LightweightAgentResult(
                        scored_cards=[],
                        feedback_for_query_agent=f"Batch {batch_info[0] + 1} failed to evaluate"
                    )
                else:
                    raise e  # Re-raise for bulk evaluation
    
    async def evaluate_cards(
        self,
        natural_language_request: str,
        cards: List[Card],
        iteration_count: int,
        previous_queries: List[str] = None,
        use_streaming: bool = False,
        total_cards: int = None
    ) -> LightweightEvaluationResult:
        """
        Evaluate how relevant the found cards are to the original request
        
        Args:
            natural_language_request: The original user request
            cards: List of cards found by search
            iteration_count: Current iteration number
            previous_queries: List of queries that have been tried
            total_cards: Total number of cards found in the search (for batch context)
            
        Returns:
            LightweightEvaluationResult with scored cards and feedback
        """
        previous_queries = previous_queries or []
        
        # Choose evaluation method based on configuration
        if ENABLE_PARALLEL_EVALUATION and len(cards) > EVALUATION_BATCH_SIZE:
            print(f"🚀 Using parallel batch evaluation (batch size: {EVALUATION_BATCH_SIZE})")
            return await self.evaluate_cards_parallel(
                natural_language_request=natural_language_request,
                cards=cards,
                iteration_count=iteration_count,
                previous_queries=previous_queries,
                use_streaming=use_streaming,
                total_cards=total_cards or len(cards)
            )
        else:
            # Fall back to original bulk evaluation for smaller card sets
            if ENABLE_PARALLEL_EVALUATION:
                print(f"📊 Using bulk evaluation for {len(cards)} cards (below batch size threshold)")
            else:
                print(f"📊 Using bulk evaluation (parallel processing disabled)")
            return await self._evaluate_cards_bulk(
                natural_language_request=natural_language_request,
                cards=cards,
                iteration_count=iteration_count,
                previous_queries=previous_queries,
                use_streaming=use_streaming,
                total_cards=total_cards or len(cards)
            )

    async def _evaluate_cards_bulk(
        self,
        natural_language_request: str,
        cards: List[Card],
        iteration_count: int,
        previous_queries: List[str] = None,
        use_streaming: bool = False,
        total_cards: int = None
    ) -> LightweightEvaluationResult:
        """
        Simplified bulk evaluation method using helper functions
        """
        total_cards = total_cards or len(cards)
        
        # Build prompt using helper
        prompt = self._build_evaluation_prompt(
            natural_language_request=natural_language_request,
            cards=cards,
            iteration_count=iteration_count,
            total_cards=total_cards,
            previous_queries=previous_queries
        )
        
        # Execute agent using helper
        agent_result = await self._execute_agent(
            prompt=prompt,
            use_streaming=use_streaming,
            cards_count=len(cards)
        )
        
        # Calculate metrics using helper
        avg_score, should_continue = self._calculate_metrics(
            agent_result=agent_result,
            total_cards=total_cards,
            iteration_count=iteration_count
        )
        
        # Create and return result using helper
        return self._create_evaluation_result(
            agent_result=agent_result,
            avg_score=avg_score,
            should_continue=should_continue,
            iteration_count=iteration_count
        )

    async def evaluate_cards_batch(
        self,
        natural_language_request: str,
        card_batch: List[Card],
        batch_index: int,
        total_batches: int,
        total_cards: int,
        iteration_count: int,
        previous_queries: List[str] = None,
        use_streaming: bool = False
    ) -> LightweightEvaluationResult:
        """
        Simplified batch evaluation method using helper functions
        """
        # Build prompt using helper with batch info
        prompt = self._build_evaluation_prompt(
            natural_language_request=natural_language_request,
            cards=card_batch,
            iteration_count=iteration_count,
            total_cards=total_cards,
            previous_queries=previous_queries,
            batch_info=(batch_index, total_batches)
        )
        
        # Execute agent using helper with batch info
        agent_result = await self._execute_agent(
            prompt=prompt,
            use_streaming=use_streaming,
            cards_count=len(card_batch),
            batch_info=(batch_index, total_batches)
        )
        
        # Calculate metrics using helper
        avg_score, should_continue = self._calculate_metrics(
            agent_result=agent_result,
            total_cards=total_cards,
            iteration_count=iteration_count
        )
        
        # Create and return result using helper
        return self._create_evaluation_result(
            agent_result=agent_result,
            avg_score=avg_score,
            should_continue=should_continue,
            iteration_count=iteration_count
        )

    async def evaluate_cards_parallel(
        self,
        natural_language_request: str,
        cards: List[Card],
        iteration_count: int,
        previous_queries: List[str] = None,
        use_streaming: bool = False,
        total_cards: int = None
    ) -> LightweightEvaluationResult:
        """
        Simplified parallel evaluation method using helper functions
        """
        batch_size = EVALUATION_BATCH_SIZE
        batches = [cards[i:i + batch_size] for i in range(0, len(cards), batch_size)]
        total_batches = len(batches)
        total_cards = total_cards or len(cards)
        
        print(f"🚀 Parallel evaluation: {len(cards)} cards in {total_batches} batches of {batch_size}")
        
        # Create tasks for parallel execution
        start_time = time.time()
        tasks = [
            self.evaluate_cards_batch(
                natural_language_request=natural_language_request,
                card_batch=batch,
                batch_index=i,
                total_batches=total_batches,
                total_cards=total_cards,
                iteration_count=iteration_count,
                previous_queries=previous_queries,
                use_streaming=use_streaming
            )
            for i, batch in enumerate(batches)
        ]
        
        # Execute all batches concurrently and show performance metrics
        batch_results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        print(f"⚡ All {total_batches} batches completed in {elapsed:.1f}s")
        
        # Calculate estimated sequential time for comparison
        estimated_sequential = elapsed * total_batches
        time_saved = estimated_sequential - elapsed
        if time_saved > 0:
            print(f"🚀 Estimated time saved: {time_saved:.1f}s (vs ~{estimated_sequential:.1f}s sequential)")
        
        # Combine batch results using helper
        return self._combine_batch_results(
            batch_results=batch_results,
            total_cards=total_cards,
            iteration_count=iteration_count
        )