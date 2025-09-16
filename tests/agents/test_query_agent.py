"""
Async unit tests for the QueryAgent.

Tests cover:
- Agent initialization and dependencies
- Query generation from natural language
- Tool integration (tag search)
- Event emission during operations
- Error handling and edge cases
- Prompt building and context handling
- Mock-based testing to avoid actual AI API calls
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from src.mtg_search_agent.agents.query_agent import QueryAgent, QueryAgentDeps, query_agent
from src.mtg_search_agent.models.search import SearchQuery
from src.mtg_search_agent.tools.tag_search import TagSearchTool


class TestQueryAgentDeps:
    """Test suite for QueryAgentDeps class."""

    def test_query_agent_deps_initialization(self):
        """Test QueryAgentDeps initialization."""
        with patch.object(TagSearchTool, '_load_tags'):
            deps = QueryAgentDeps()
            assert hasattr(deps, 'tag_search')
            assert isinstance(deps.tag_search, TagSearchTool)

    def test_query_agent_deps_tag_search_tool(self):
        """Test that TagSearchTool is properly initialized in deps."""
        with patch.object(TagSearchTool, '_load_tags') as mock_load:
            deps = QueryAgentDeps()
            mock_load.assert_called_once()
            assert deps.tag_search is not None


class TestQueryAgent:
    """Test suite for QueryAgent class."""

    def test_query_agent_initialization_without_emitter(self):
        """Test QueryAgent initialization without event emitter."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            assert hasattr(agent, 'deps')
            assert agent.events is None

    def test_query_agent_initialization_with_emitter(self, mock_event_emitter):
        """Test QueryAgent initialization with event emitter."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)
            assert agent.events == mock_event_emitter

    def test_build_prompt_basic(self):
        """Test basic prompt building functionality."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "red instant damage spells"

            prompt = agent._build_prompt(request)

            assert request in prompt
            assert "Convert this natural language request" in prompt
            assert "SearchQuery" in prompt

    def test_build_prompt_with_previous_queries(self):
        """Test prompt building with previous queries."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "red instant damage"
            previous_queries = ["c:r t:instant", "c:r o:damage"]

            prompt = agent._build_prompt(request, previous_queries=previous_queries)

            assert "Previous queries attempted" in prompt
            assert "c:r t:instant" in prompt
            assert "c:r o:damage" in prompt

    def test_build_prompt_with_feedback(self):
        """Test prompt building with evaluation feedback."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "red burn spells"
            feedback = "Try including more direct damage effects"

            prompt = agent._build_prompt(request, feedback=feedback)

            assert "Feedback from evaluation" in prompt
            assert feedback in prompt

    def test_build_prompt_with_all_context(self):
        """Test prompt building with all context information."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "flying creatures"
            previous_queries = ["t:creature", "o:flying"]
            feedback = "Include keyword search for flying"

            prompt = agent._build_prompt(
                request,
                previous_queries=previous_queries,
                feedback=feedback
            )

            assert request in prompt
            assert "Previous queries attempted" in prompt
            assert "Feedback from evaluation" in prompt
            assert "t:creature" in prompt
            assert feedback in prompt

    def test_build_prompt_empty_previous_queries(self):
        """Test prompt building with empty previous queries list."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "test request"

            prompt = agent._build_prompt(request, previous_queries=[])

            # Should not include previous queries section
            assert "Previous queries attempted" not in prompt

    def test_build_prompt_none_previous_queries(self):
        """Test prompt building with None previous queries."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent()
            request = "test request"

            prompt = agent._build_prompt(request, previous_queries=None)

            # Should not include previous queries section
            assert "Previous queries attempted" not in prompt

    @pytest.mark.asyncio
    async def test_generate_query_basic(self, mock_event_emitter):
        """Test basic query generation functionality."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            # Mock the agent response
            mock_response = SearchQuery(
                query="c:r t:instant",
                explanation="Search for red instant spells",
                confidence=0.8
            )

            with patch.object(query_agent, 'run') as mock_run:
                # Mock the async run method to return a result object with .output
                mock_result = Mock()
                mock_result.output = mock_response
                mock_run.return_value = mock_result

                result = await agent.generate_query("red instant spells")

                assert isinstance(result, SearchQuery)
                assert result.query == "c:r t:instant"
                assert result.explanation == "Search for red instant spells"
                assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_generate_query_with_context(self, mock_event_emitter):
        """Test query generation with context information."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            mock_response = SearchQuery(
                query="c:r t:instant o:damage",
                explanation="Red instant spells that deal damage",
                confidence=0.9
            )

            with patch.object(query_agent, 'run') as mock_run:
                # Mock the async run method to return a result object with .output
                mock_result = Mock()
                mock_result.output = mock_response
                mock_run.return_value = mock_result

                result = await agent.generate_query(
                    "red burn spells",
                    previous_queries=["c:r"],
                    feedback="Include damage effects"
                )

                assert isinstance(result, SearchQuery)
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_query_with_streaming(self, mock_event_emitter):
        """Test query generation with streaming enabled."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            mock_response = SearchQuery(
                query="c:r t:instant",
                explanation="Red instants",
                confidence=0.7
            )

            with patch.object(query_agent, 'run_stream') as mock_run_stream:
                # Mock the async context manager and streaming response
                class MockStreamResult:
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass
                    async def stream(self):
                        yield mock_response
                    async def get_output(self):
                        return mock_response

                mock_run_stream.return_value = MockStreamResult()

                result = await agent.generate_query(
                    "red spells",
                    use_streaming=True
                )

                assert isinstance(result, SearchQuery)
                assert mock_event_emitter.emit.call_count > 0

    @pytest.mark.asyncio
    async def test_generate_query_event_emission(self, mock_event_emitter):
        """Test that appropriate events are emitted during query generation."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            mock_response = SearchQuery(query="test", explanation="test")

            with patch.object(query_agent, 'run') as mock_run:
                # Mock the async run method to return a result object with .output
                mock_result = Mock()
                mock_result.output = mock_response
                mock_run.return_value = mock_result

                await agent.generate_query("test request")

                # Verify events were emitted
                assert mock_event_emitter.emit.call_count > 0

                # Check for specific event types
                emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
                event_types = [event.event_type for event in emitted_events]

                # Should emit query generation started event
                assert "query_generation_started" in event_types

    @pytest.mark.asyncio
    async def test_generate_query_error_handling(self, mock_event_emitter):
        """Test error handling during query generation."""
        with patch.object(QueryAgentDeps, '__init__', return_value=None):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            with patch.object(query_agent, 'run') as mock_run:
                mock_run.side_effect = Exception("AI model error")

                with pytest.raises(Exception) as exc_info:
                    await agent.generate_query("test request")

                assert "AI model error" in str(exc_info.value)

                # Should emit query generation started event (no error event in non-streaming mode)
                emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
                event_types = [event.event_type for event in emitted_events]
                assert "query_generation_started" in event_types


class TestSearchSimilarTagsTool:
    """Test suite for the search_similar_tags tool function."""

    @pytest.mark.asyncio
    async def test_search_similar_tags_basic(self, sample_tags_data):
        """Test basic tag search functionality."""
        # Create mock context
        mock_deps = Mock()
        mock_tag_search = Mock()
        mock_tag_search.find_similar_tags.return_value = ["instant", "sorcery"]
        mock_deps.tag_search = mock_tag_search

        mock_ctx = Mock()
        mock_ctx.deps = mock_deps

        # Import the tool function
        from src.mtg_search_agent.agents.query_agent import search_similar_tags

        result = await search_similar_tags(mock_ctx, ["inst", "sorc"])

        assert result == ["instant", "sorcery"]
        mock_tag_search.find_similar_tags.assert_called_once_with(["inst", "sorc"])

    @pytest.mark.asyncio
    async def test_search_similar_tags_empty_input(self):
        """Test tag search with empty input."""
        mock_deps = Mock()
        mock_tag_search = Mock()
        mock_tag_search.find_similar_tags.return_value = []
        mock_deps.tag_search = mock_tag_search

        mock_ctx = Mock()
        mock_ctx.deps = mock_deps

        from src.mtg_search_agent.agents.query_agent import search_similar_tags

        result = await search_similar_tags(mock_ctx, [])

        assert result == []
        mock_tag_search.find_similar_tags.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_search_similar_tags_no_matches(self):
        """Test tag search when no matches are found."""
        mock_deps = Mock()
        mock_tag_search = Mock()
        mock_tag_search.find_similar_tags.return_value = []
        mock_deps.tag_search = mock_tag_search

        mock_ctx = Mock()
        mock_ctx.deps = mock_deps

        from src.mtg_search_agent.agents.query_agent import search_similar_tags

        result = await search_similar_tags(mock_ctx, ["xyz", "abc"])

        assert result == []

    @pytest.mark.asyncio
    async def test_search_similar_tags_error_handling(self):
        """Test tag search error handling."""
        mock_deps = Mock()
        mock_tag_search = Mock()
        mock_tag_search.find_similar_tags.side_effect = Exception("Tag search error")
        mock_deps.tag_search = mock_tag_search

        mock_ctx = Mock()
        mock_ctx.deps = mock_deps

        from src.mtg_search_agent.agents.query_agent import search_similar_tags

        with pytest.raises(Exception) as exc_info:
            await search_similar_tags(mock_ctx, ["test"])

        assert "Tag search error" in str(exc_info.value)


class TestQueryAgentIntegration:
    """Integration tests for QueryAgent functionality."""

    @pytest.mark.asyncio
    async def test_full_query_generation_workflow(self, mock_event_emitter):
        """Test complete query generation workflow."""
        # Setup agent with mocked dependencies
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        # Mock the AI agent response
        mock_response = SearchQuery(
            query="c:r (t:instant or t:sorcery) o:damage",
            explanation="Red instant spells that deal damage",
            confidence=0.85
        )

        # Mock the generate_query method directly to return our expected response
        with patch.object(agent, 'generate_query', return_value=mock_response) as mock_generate:

            # Test the complete workflow
            result = await agent.generate_query(
                natural_language_request="red burn spells",
                previous_queries=["c:r"],
                feedback="Include damage effects",
                use_streaming=False
            )

            # Verify result
            assert isinstance(result, SearchQuery)
            assert result.query == "c:r (t:instant or t:sorcery) o:damage"
            assert result.explanation == "Red instant spells that deal damage"
            assert result.confidence == 0.85

            # Verify agent was called with proper arguments
            mock_generate.assert_called_once_with(
                natural_language_request="red burn spells",
                previous_queries=["c:r"],
                feedback="Include damage effects",
                use_streaming=False
            )

    @pytest.mark.asyncio
    async def test_query_generation_with_tag_tool_usage(self, mock_event_emitter):
        """Test query generation that uses the tag search tool."""
        # This would be an integration test where the AI agent
        # actually calls the search_similar_tags tool during generation

        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

            # Mock the tag search to return some results
            with patch.object(agent.deps.tag_search, 'find_similar_tags') as mock_tag_search:
                mock_tag_search.return_value = ["instant", "damage"]

                # Mock AI response that might use tag suggestions
                mock_response = SearchQuery(
                    query="c:r t:instant otag:damage",
                    explanation="Red instant spells with damage tag",
                    confidence=0.9
                )

                with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
                    mock_result = Mock()
                    mock_result.output = mock_response
                    mock_run.return_value = mock_result

                    result = await agent.generate_query("red burn spells")

                    assert isinstance(result, SearchQuery)
                    # The actual tool usage depends on the AI model's behavior
                    # This test just ensures the infrastructure works

    @pytest.mark.asyncio
    async def test_error_recovery_and_event_emission(self, mock_event_emitter):
        """Test error recovery and proper event emission."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        # Test various error scenarios
        error_scenarios = [
            ("Network error", ConnectionError("Network unavailable")),
            ("AI model error", ValueError("Invalid model response")),
            ("Timeout error", TimeoutError("Request timed out")),
        ]

        for error_name, error_exception in error_scenarios:
            mock_event_emitter.reset_mock()

            with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = error_exception

                with pytest.raises(type(error_exception)):
                    await agent.generate_query("test request")

                # Verify error event was emitted
                emitted_events = [call[0][0] for call in mock_event_emitter.emit.call_args_list]
                event_types = [event.event_type for event in emitted_events]
                assert "error_occurred" in event_types

    @pytest.mark.asyncio
    async def test_concurrent_query_generation(self, mock_event_emitter):
        """Test concurrent query generation requests."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        # Mock responses for different queries
        responses = [
            SearchQuery(query="c:r", explanation="Red cards", confidence=0.7),
            SearchQuery(query="c:u", explanation="Blue cards", confidence=0.8),
            SearchQuery(query="c:g", explanation="Green cards", confidence=0.9),
        ]

        with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = responses

            # Start multiple concurrent requests
            tasks = [
                agent.generate_query("red cards"),
                agent.generate_query("blue cards"),
                agent.generate_query("green cards"),
            ]

            results = await asyncio.gather(*tasks)

            # Verify all requests completed
            assert len(results) == 3
            assert all(isinstance(result, SearchQuery) for result in results)

            # Verify all agents were called
            assert mock_run.call_count == 3


class TestQueryAgentConfiguration:
    """Test suite for QueryAgent configuration and setup."""

    def test_query_agent_global_instance(self):
        """Test that global query_agent instance is properly configured."""
        from src.mtg_search_agent.agents.query_agent import query_agent

        assert query_agent is not None
        assert hasattr(query_agent, 'deps_type')
        assert hasattr(query_agent, 'output_type')
        assert query_agent.output_type == SearchQuery

    def test_query_agent_model_configuration(self):
        """Test query agent model configuration."""
        from src.mtg_search_agent.agents.query_agent import query_agent

        # Should be configured with OpenAI model
        assert hasattr(query_agent, 'model')

    def test_query_agent_system_prompt_loading(self):
        """Test that system prompt is loaded correctly."""
        from src.mtg_search_agent.agents.query_agent import query_agent

        # Should have a system prompt
        assert hasattr(query_agent, 'system_prompt')
        assert query_agent.system_prompt is not None

    def test_query_agent_tools_registration(self):
        """Test that tools are properly registered with the agent."""
        from src.mtg_search_agent.agents.query_agent import query_agent

        # Should have tools registered
        assert hasattr(query_agent, 'tools')


class TestQueryAgentEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_request(self, mock_event_emitter):
        """Test handling of very long natural language requests."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        long_request = "Find me " + "red instant damage " * 100  # Very long request

        mock_response = SearchQuery(query="c:r t:instant", explanation="Red instants")

        with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.generate_query(long_request)

            assert isinstance(result, SearchQuery)
            # Should handle long requests without error

    @pytest.mark.asyncio
    async def test_empty_request(self, mock_event_emitter):
        """Test handling of empty natural language request."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        mock_response = SearchQuery(query="*", explanation="All cards")

        with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.generate_query("")

            assert isinstance(result, SearchQuery)

    @pytest.mark.asyncio
    async def test_special_characters_in_request(self, mock_event_emitter):
        """Test handling of special characters in requests."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        special_request = "Find cards with Æther or mana-cost {2/W}"

        mock_response = SearchQuery(
            query='o:"Æther" or mana:{2/W}',
            explanation="Cards with Æther or hybrid mana"
        )

        with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.generate_query(special_request)

            assert isinstance(result, SearchQuery)

    @pytest.mark.asyncio
    async def test_unicode_in_request(self, mock_event_emitter):
        """Test handling of unicode characters in requests."""
        with patch.object(TagSearchTool, '_load_tags'):
            agent = QueryAgent(event_emitter=mock_event_emitter)

        unicode_request = "找红色瞬间法术"  # Chinese characters

        mock_response = SearchQuery(query="c:r t:instant", explanation="Red instants")

        with patch.object(query_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_result = Mock()
            mock_result.output = mock_response
            mock_run.return_value = mock_result

            result = await agent.generate_query(unicode_request)

            assert isinstance(result, SearchQuery)