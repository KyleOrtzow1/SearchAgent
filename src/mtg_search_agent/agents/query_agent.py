from typing import List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel
import sys
import os
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()


from ..models.search import SearchQuery, TagSuggestion
from ..tools.tag_search import TagSearchTool

# Import new event classes
from ..events import (
    QueryGenerationStartedEvent,
    QueryStreamingProgressEvent,
    ErrorOccurredEvent,
)

# Constants for better maintainability
QUERY_TRUNCATE_LENGTH = 60
EXPLANATION_TRUNCATE_LENGTH = 40

from ..prompts import load_query_agent_prompt


class QueryAgentDeps:
    """Dependencies for the Query Agent"""
    def __init__(self):
        self.tag_search = TagSearchTool()


# Create the query agent with comprehensive Scryfall syntax knowledge
query_agent = Agent(
    model=OpenAIResponsesModel('gpt-5-mini'),
    deps_type=QueryAgentDeps,
    output_type=SearchQuery,
    system_prompt=load_query_agent_prompt()
)


@query_agent.tool
async def search_similar_tags(ctx: RunContext[QueryAgentDeps], guess_tags: List[str]) -> List[str]:
    """
    Find tags similar to your guesses using fuzzy matching.
    
    Args:
        guess_tags: List of tag guesses to find similar matches for
        
    Returns:
        List of valid tag names (strings only) sorted by relevance
    """
    return ctx.deps.tag_search.find_similar_tags(guess_tags)




class QueryAgent:
    """Agent responsible for converting natural language to Scryfall queries"""
    
    def __init__(self, event_emitter=None):
        self.deps = QueryAgentDeps()
        self.events = event_emitter
    
    def _build_prompt(
        self, 
        natural_language_request: str, 
        previous_queries: List[str] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Build the prompt for query generation with context
        
        Args:
            natural_language_request: The user's natural language request
            previous_queries: List of previously attempted queries
            feedback: Feedback from evaluation agent on how to improve
            
        Returns:
            Complete prompt string
        """
        previous_queries = previous_queries or []
        
        prompt_parts = [
            f"Convert this natural language request to a Scryfall search query: {natural_language_request}"
        ]
        
        if previous_queries:
            prompt_parts.append(f"Previous queries attempted: {', '.join(previous_queries)}")
        
        if feedback:
            prompt_parts.append(f"Feedback from evaluation: {feedback}")
        
        prompt_parts.append(
            "Use your Scryfall knowledge to create the query. "
            "Consider if the request involves functional categories that might benefit from tags - "
            "if so, use the search_similar_tags tool to find relevant tags and include them with otag: format. "
            "Return a SearchQuery with the final query string and explanation."
        )
        
        return "\n\n".join(prompt_parts)
    
    def _display_streaming_progress(self, partial_query) -> None:
        """
        Display streaming progress for query generation
        
        Args:
            partial_query: Partial query object with query and explanation attributes
        """
        if self.events:
            # Emit streaming progress event using new API
            query = getattr(partial_query, 'query', '') or ''
            explanation = getattr(partial_query, 'explanation', '') or ''
            self.events.emit(QueryStreamingProgressEvent(query, explanation))
    
    async def _execute_with_streaming(self, prompt: str) -> SearchQuery:
        """
        Execute query generation with streaming enabled
        
        Args:
            prompt: Complete prompt string
            
        Returns:
            SearchQuery object with Scryfall syntax
        """
        # Emit query generation started event
        if self.events:
            self.events.emit(QueryGenerationStartedEvent("", 0))  # No request/iteration context here
        
        try:
            async with query_agent.run_stream(prompt, deps=self.deps) as result:
                # Stream structured output as it's being built
                async for partial_query in result.stream():
                    self._display_streaming_progress(partial_query)
            
            return await result.get_output()
            
        except Exception as e:
            # Emit error event instead of printing
            if self.events:
                self.events.emit(ErrorOccurredEvent("query_streaming_error", str(e), {"fallback": "non_streaming"}))
            return await self._execute_without_streaming(prompt)
    
    async def _execute_without_streaming(self, prompt: str) -> SearchQuery:
        """
        Execute query generation without streaming
        
        Args:
            prompt: Complete prompt string
            
        Returns:
            SearchQuery object with Scryfall syntax
        """
        # Emit query generation started event
        if self.events:
            self.events.emit(QueryGenerationStartedEvent("", 0))  # No request/iteration context here
        
        result = await query_agent.run(prompt, deps=self.deps)
        return result.output
    
    async def generate_query(
        self, 
        natural_language_request: str, 
        previous_queries: List[str] = None,
        feedback: Optional[str] = None,
        use_streaming: bool = False
    ) -> SearchQuery:
        """
        Generate a Scryfall query from natural language request
        
        Args:
            natural_language_request: The user's natural language request
            previous_queries: List of previously attempted queries
            feedback: Feedback from evaluation agent on how to improve
            use_streaming: Whether to use streaming mode for real-time progress
            
        Returns:
            SearchQuery object with Scryfall syntax
        """
        prompt = self._build_prompt(natural_language_request, previous_queries, feedback)
        
        if use_streaming:
            return await self._execute_with_streaming(prompt)
        else:
            return await self._execute_without_streaming(prompt)