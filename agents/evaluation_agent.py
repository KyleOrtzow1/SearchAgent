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
        Original bulk evaluation method (renamed from evaluate_cards)
        """
        total_cards = total_cards or len(cards)
        previous_queries = previous_queries or []
        
        # Build the evaluation prompt
        card_summaries = []
        for i, card in enumerate(cards, 1):
            card_info = f"{i}. {card.name} ({card.mana_cost or 'No cost'}) - {card.type_line}"
            if card.oracle_text:
                card_info += f" | {card.oracle_text}"
            card_summaries.append(card_info)
        
        prompt_parts = [
            f"Original request: {natural_language_request}",
            "",
            "Cards found:",
        ]
        prompt_parts.extend(card_summaries)
        
        if previous_queries:
            prompt_parts.extend([
                "",
                f"Previous queries tried: {', '.join(previous_queries)}"
            ])
        
        prompt_parts.extend([
            "",
            f"This is iteration {iteration_count} of the search process.",
            "",
            "Evaluate each card's relevance to the original request on a scale of 1-10.",
            "Score 1-3: Not relevant, Score 4-6: Somewhat relevant, Score 7-10: Highly relevant",
            "",
            "Return a LightweightAgentResult with:",
            "- scored_cards: List with ONLY name, score, and reasoning for each card", 
            f"- feedback_for_query_agent: If scores are low OR fewer than {TOP_CARDS_TO_DISPLAY} total cards found (currently {total_cards}), provide specific suggestions for broader search terms or different criteria",
            "",
            "DO NOT include full card data - just name, score, and reasoning!"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Use either streaming or direct generation based on parameter
        if use_streaming:
            print("📊 Evaluation Agent analyzing cards...")
            print(f"   Evaluating {len(cards)} cards for relevance...")
            print("🧠 ", end="", flush=True)
            
            # Use run_stream() with native structured streaming
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
                                print(f"\r🧠 Evaluated {cards_evaluated}/{len(cards)} cards", end="", flush=True)
                        
                        # Show average score as it develops
                        if hasattr(partial_evaluation, 'average_score') and partial_evaluation.average_score > 0:
                            print(f" | Avg Score: {partial_evaluation.average_score:.1f}", end="", flush=True)
                
                print(f"\n📈 Card evaluation complete!")
                agent_result = await result.get_output()
                
                # Calculate average score from individual card scores
                scores = [card.score for card in agent_result.scored_cards]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Check if we need to continue searching based on score OR insufficient total card count
                has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
                should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
                
                # Create full evaluation result with calculated fields
                return LightweightEvaluationResult(
                    scored_cards=agent_result.scored_cards,
                    average_score=avg_score,
                    should_continue=should_continue,
                    feedback_for_query_agent=agent_result.feedback_for_query_agent,
                    iteration_count=iteration_count
                )
                
            except Exception as e:
                print(f"\n⚠️ Streaming error: {e}")
                print("🔄 Falling back to non-streaming mode...")
                result = await lightweight_evaluation_agent.run(prompt)
                agent_result = result.output
                
                # Calculate average score from individual card scores
                scores = [card.score for card in agent_result.scored_cards]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Check if we need to continue searching based on score OR insufficient total card count
                has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
                should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
                
                # Create full evaluation result with calculated fields
                return LightweightEvaluationResult(
                    scored_cards=agent_result.scored_cards,
                    average_score=avg_score,
                    should_continue=should_continue,
                    feedback_for_query_agent=agent_result.feedback_for_query_agent,
                    iteration_count=iteration_count
                )
        else:
            # Use direct generation without streaming to save tokens
            print("📊 Evaluation Agent analyzing cards...")
            print(f"   Evaluating {len(cards)} cards for relevance...")
            result = await lightweight_evaluation_agent.run(prompt)
            agent_result = result.output
            
            # Calculate average score from individual card scores
            scores = [card.score for card in agent_result.scored_cards]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            print("📈 Card evaluation complete!")
            # Check if we need to continue searching based on score OR insufficient total card count
            has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
            should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
            
            # Create full evaluation result with calculated fields
            return LightweightEvaluationResult(
                scored_cards=agent_result.scored_cards,
                average_score=avg_score,
                should_continue=should_continue,
                feedback_for_query_agent=agent_result.feedback_for_query_agent,
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
        Evaluate a single batch of cards
        
        Args:
            natural_language_request: The original user request
            card_batch: Batch of cards to evaluate
            batch_index: Index of this batch (0-based)
            total_batches: Total number of batches
            total_cards: Total number of cards found in the search
            iteration_count: Current iteration number
            previous_queries: List of queries that have been tried
            
        Returns:
            LightweightEvaluationResult for this batch
        """
        previous_queries = previous_queries or []
        
        # Build the evaluation prompt for this batch
        card_summaries = []
        for i, card in enumerate(card_batch, 1):
            card_info = f"{i}. {card.name} ({card.mana_cost or 'No cost'}) - {card.type_line}"
            if card.oracle_text:
                card_info += f" | {card.oracle_text}"
            card_summaries.append(card_info)
        
        prompt_parts = [
            f"Original request: {natural_language_request}",
            "",
            f"Cards to evaluate (Batch {batch_index + 1}/{total_batches} - showing {len(card_batch)} of {total_cards} total cards found):",
        ]
        prompt_parts.extend(card_summaries)
        
        if previous_queries:
            prompt_parts.extend([
                "",
                f"Previous queries tried: {', '.join(previous_queries)}"
            ])
        
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
            "",
            "CRITICAL: Every scored_card MUST have exactly these fields:",
            "  - name: string (card name)",
            "  - score: integer between 1 and 10",  
            "  - reasoning: string (why this score)",
            "",
            "DO NOT include full card data - just name, score, and reasoning!"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Use either streaming or direct generation based on parameter
        if use_streaming:
            print(f"🧠 Batch {batch_index + 1}/{total_batches} evaluating {len(card_batch)} cards...", end="", flush=True)
            
            try:
                async with lightweight_evaluation_agent.run_stream(prompt) as result:
                    # Stream structured output as it's being built
                    cards_evaluated = 0
                    async for partial_evaluation in result.stream():
                        # Show evaluation progress for this batch
                        if hasattr(partial_evaluation, 'scored_cards'):
                            current_cards = len(partial_evaluation.scored_cards)
                            if current_cards > cards_evaluated:
                                cards_evaluated = current_cards
                                print(f"\r🧠 Batch {batch_index + 1}/{total_batches}: {cards_evaluated}/{len(card_batch)} cards", end="", flush=True)
                        
                        # Show average score as it develops
                        if hasattr(partial_evaluation, 'average_score') and partial_evaluation.average_score > 0:
                            print(f" | Score: {partial_evaluation.average_score:.1f}", end="", flush=True)
                
                print(" ✅")
                agent_result = await result.get_output()
                
                # Calculate average score from individual card scores
                scores = [card.score for card in agent_result.scored_cards]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Check if we need to continue searching based on score OR insufficient total card count
                has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
                should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
                
                # Create full evaluation result with calculated fields
                return LightweightEvaluationResult(
                    scored_cards=agent_result.scored_cards,
                    average_score=avg_score,
                    should_continue=should_continue,
                    feedback_for_query_agent=agent_result.feedback_for_query_agent,
                    iteration_count=iteration_count
                )
                
            except Exception as e:
                print(f" ⚠️ Error: {e}")
                print(f"🔄 Batch {batch_index + 1} falling back to non-streaming...")
                try:
                    result = await lightweight_evaluation_agent.run(prompt)
                    agent_result = result.output
                    
                    # Calculate average score from individual card scores
                    scores = [card.score for card in agent_result.scored_cards]
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    
                    # Check if we need to continue searching based on score OR insufficient total card count
                    has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
                    should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
                    
                    # Create full evaluation result with calculated fields
                    return LightweightEvaluationResult(
                        scored_cards=agent_result.scored_cards,
                        average_score=avg_score,
                        should_continue=should_continue,
                        feedback_for_query_agent=agent_result.feedback_for_query_agent,
                        iteration_count=iteration_count
                    )
                except Exception as fallback_error:
                    print(f" ❌ Fallback failed: {fallback_error}")
                    # Return empty result for this batch to prevent total failure
                    return LightweightEvaluationResult(
                        scored_cards=[],
                        average_score=0.0,
                        should_continue=True,
                        feedback_for_query_agent=f"Batch {batch_index + 1} failed to evaluate",
                        iteration_count=iteration_count
                    )
        else:
            # Use direct generation without streaming
            print(f"🧠 Batch {batch_index + 1}/{total_batches} evaluating {len(card_batch)} cards...", end="", flush=True)
            try:
                result = await lightweight_evaluation_agent.run(prompt)
                agent_result = result.output
                
                # Calculate average score from individual card scores
                scores = [card.score for card in agent_result.scored_cards]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                print(" ✅")
                # Check if we need to continue searching based on score OR insufficient total card count
                has_insufficient_cards = total_cards < TOP_CARDS_TO_DISPLAY
                should_continue = (avg_score < 6 or has_insufficient_cards) and iteration_count < 5
                
                # Create full evaluation result with calculated fields
                return LightweightEvaluationResult(
                    scored_cards=agent_result.scored_cards,
                    average_score=avg_score,
                    should_continue=should_continue,
                    feedback_for_query_agent=agent_result.feedback_for_query_agent,
                    iteration_count=iteration_count
                )
            except Exception as e:
                print(f" ❌ Error: {e}")
                # Return empty result for this batch to prevent total failure
                return LightweightEvaluationResult(
                    scored_cards=[],
                    average_score=0.0,
                    should_continue=True,
                    feedback_for_query_agent=f"Batch {batch_index + 1} failed to evaluate",
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
        Evaluate cards using parallel batch processing
        
        Args:
            natural_language_request: The original user request
            cards: List of cards found by search
            iteration_count: Current iteration number
            previous_queries: List of queries that have been tried
            total_cards: Total number of cards found in the search
            
        Returns:
            Combined LightweightEvaluationResult from all batches
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
        
        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        print(f"⚡ All {total_batches} batches completed in {elapsed:.1f}s")
        
        # Calculate estimated sequential time for comparison
        estimated_sequential = elapsed * total_batches
        time_saved = estimated_sequential - elapsed
        if time_saved > 0:
            print(f"🚀 Estimated time saved: {time_saved:.1f}s (vs ~{estimated_sequential:.1f}s sequential)")
        
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