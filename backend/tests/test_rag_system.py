"""
Tests for RAGSystem - query orchestration and session management.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @pytest.fixture
    def rag_system(self, mock_config):
        """RAGSystem with all dependencies mocked"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class:

            # Setup mock instances
            mock_vector_store = Mock()
            mock_ai_generator = Mock()
            mock_session_manager = Mock()

            mock_vs_class.return_value = mock_vector_store
            mock_ai_class.return_value = mock_ai_generator
            mock_sm_class.return_value = mock_session_manager

            # Default behaviors
            mock_ai_generator.generate_response.return_value = "AI response"
            mock_session_manager.get_conversation_history.return_value = None
            mock_vector_store.get_lesson_link.return_value = None
            mock_vector_store.search.return_value = SearchResults(
                documents=[], metadata=[], distances=[]
            )

            system = RAGSystem(mock_config)
            system._mock_ai = mock_ai_generator
            system._mock_session = mock_session_manager
            system._mock_vector = mock_vector_store

            return system

    # ===== Test Case 1: Full query flow =====
    def test_query_returns_response_and_sources_tuple(self, rag_system):
        """
        Given: RAGSystem is properly initialized
        When: query() is called
        Then: Returns tuple of (response_string, sources_list)
        """
        result = rag_system.query("What is MCP?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_passes_tools_to_ai_generator(self, rag_system):
        """
        Given: RAGSystem with registered tools
        When: query() is called
        Then: Tool definitions are passed to AIGenerator
        """
        rag_system.query("Search question")

        call_args = rag_system._mock_ai.generate_response.call_args
        assert 'tools' in call_args.kwargs
        assert call_args.kwargs['tools'] is not None
        assert len(call_args.kwargs['tools']) == 2  # search + outline tools

    def test_query_passes_tool_manager_to_ai_generator(self, rag_system):
        """
        Given: RAGSystem with tool_manager
        When: query() is called
        Then: tool_manager is passed to AIGenerator
        """
        rag_system.query("Query text")

        call_args = rag_system._mock_ai.generate_response.call_args
        assert 'tool_manager' in call_args.kwargs
        assert call_args.kwargs['tool_manager'] == rag_system.tool_manager

    def test_query_wraps_user_question_in_prompt(self, rag_system):
        """
        Given: User's raw question
        When: query() is called
        Then: Question is wrapped in instructional prompt
        """
        rag_system.query("What is machine learning?")

        call_args = rag_system._mock_ai.generate_response.call_args
        query_param = call_args.kwargs['query']
        assert "Answer this question about course materials" in query_param
        assert "What is machine learning?" in query_param

    # ===== Test Case 2: Query without session (no history) =====
    def test_query_without_session_id_passes_no_history(self, rag_system):
        """
        Given: Query called without session_id
        When: AIGenerator.generate_response() is called
        Then: conversation_history is None
        """
        rag_system.query("Question without session")

        call_args = rag_system._mock_ai.generate_response.call_args
        assert call_args.kwargs.get('conversation_history') is None

    def test_query_without_session_does_not_fetch_history(self, rag_system):
        """
        Given: Query called without session_id
        When: Query is processed
        Then: SessionManager.get_conversation_history() is not called
        """
        rag_system.query("No session query")

        rag_system._mock_session.get_conversation_history.assert_not_called()

    def test_query_without_session_does_not_add_to_history(self, rag_system):
        """
        Given: Query called without session_id
        When: Query completes
        Then: SessionManager.add_exchange() is not called
        """
        rag_system.query("No session query")

        rag_system._mock_session.add_exchange.assert_not_called()

    # ===== Test Case 3: Query with session (includes history) =====
    def test_query_with_session_fetches_history(self, rag_system):
        """
        Given: Query called with session_id
        When: Processing query
        Then: Fetches history from SessionManager
        """
        rag_system._mock_session.get_conversation_history.return_value = "Previous conversation"

        rag_system.query("Follow-up", session_id="session_123")

        rag_system._mock_session.get_conversation_history.assert_called_once_with("session_123")

    def test_query_with_session_passes_history_to_ai(self, rag_system):
        """
        Given: Session has conversation history
        When: query() is called with session_id
        Then: History is passed to AIGenerator
        """
        rag_system._mock_session.get_conversation_history.return_value = "User: Hi\nAssistant: Hello"

        rag_system.query("Next question", session_id="session_123")

        call_args = rag_system._mock_ai.generate_response.call_args
        assert call_args.kwargs['conversation_history'] == "User: Hi\nAssistant: Hello"

    def test_query_with_session_adds_exchange_after_response(self, rag_system):
        """
        Given: Query with session_id
        When: AI generates response
        Then: Exchange is added to session history
        """
        rag_system._mock_ai.generate_response.return_value = "The answer is X"

        rag_system.query("What is X?", session_id="session_456")

        rag_system._mock_session.add_exchange.assert_called_once()
        call_args = rag_system._mock_session.add_exchange.call_args
        assert call_args[0][0] == "session_456"
        assert "What is X?" in call_args[0][1]
        assert call_args[0][2] == "The answer is X"

    # ===== Test Case 4: Source extraction and reset =====
    def test_query_retrieves_sources_from_tool_manager(self, rag_system):
        """
        Given: Search tool has executed and set sources
        When: query() completes
        Then: Sources from tool_manager are returned
        """
        expected_sources = [{"text": "Course A", "url": "https://example.com"}]
        rag_system.tool_manager.get_last_sources = Mock(return_value=expected_sources)

        response, sources = rag_system.query("Search something")

        assert sources == expected_sources

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """
        Given: Query that triggers source collection
        When: query() completes
        Then: tool_manager.reset_sources() is called
        """
        rag_system.tool_manager.reset_sources = Mock()

        rag_system.query("Any query")

        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_returns_empty_sources_when_none_found(self, rag_system):
        """
        Given: Query that doesn't produce sources
        When: query() completes
        Then: Returns empty sources list
        """
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])

        response, sources = rag_system.query("General question")

        assert sources == []

    # ===== Test Case 5: Multiple queries in same session =====
    def test_multiple_queries_each_add_to_history(self, rag_system):
        """
        Given: Multiple queries to same session
        When: Each query completes
        Then: Each adds to session history
        """
        rag_system._mock_ai.generate_response.side_effect = ["Answer 1", "Answer 2"]

        rag_system.query("Question 1", session_id="multi_session")
        rag_system.query("Question 2", session_id="multi_session")

        assert rag_system._mock_session.add_exchange.call_count == 2

    def test_second_query_receives_first_query_history(self, rag_system):
        """
        Given: Session with previous conversation
        When: Second query is made
        Then: Previous history is passed to AI
        """
        rag_system._mock_session.get_conversation_history.side_effect = [
            None,  # First query - no history
            "User: Q1\nAssistant: A1"  # Second query - has history
        ]
        rag_system._mock_ai.generate_response.side_effect = ["A1", "A2"]

        rag_system.query("Q1", session_id="session_x")
        rag_system.query("Q2", session_id="session_x")

        second_call_args = rag_system._mock_ai.generate_response.call_args_list[1]
        assert second_call_args.kwargs['conversation_history'] == "User: Q1\nAssistant: A1"


class TestRAGSystemSourceIntegration:
    """Integration-style tests for source tracking through full flow"""

    @pytest.fixture
    def rag_with_real_tool_manager(self, mock_config):
        """RAGSystem with real ToolManager but mocked external dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager'):

            # VectorStore mock that returns proper SearchResults
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=["Content about topic"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.1]
            )
            mock_vector_store.get_lesson_link.return_value = "https://test.com/lesson1"
            mock_vector_store._resolve_course_name.return_value = "Test Course"
            mock_vs_class.return_value = mock_vector_store

            # AI that returns a response
            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Generated answer"
            mock_ai_class.return_value = mock_ai

            system = RAGSystem(mock_config)
            system._mock_ai = mock_ai
            system._mock_vs = mock_vector_store
            return system

    def test_search_tool_produces_sources_retrieved_by_query(self, rag_with_real_tool_manager):
        """
        Given: Search tool execution produces sources
        When: Sources are retrieved after query
        Then: Sources contain expected data
        """
        # Manually execute search to set sources
        rag_with_real_tool_manager.search_tool.execute(query="test query")

        sources = rag_with_real_tool_manager.tool_manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"
        assert sources[0]["url"] == "https://test.com/lesson1"

    def test_sources_cleared_between_queries(self, rag_with_real_tool_manager):
        """
        Given: Query that produces sources
        When: Sources are reset
        Then: Subsequent get_last_sources returns empty
        """
        # Execute search to set sources
        rag_with_real_tool_manager.search_tool.execute(query="first query")
        first_sources = rag_with_real_tool_manager.tool_manager.get_last_sources()
        assert len(first_sources) > 0

        # Reset sources
        rag_with_real_tool_manager.tool_manager.reset_sources()

        # Verify cleared
        assert rag_with_real_tool_manager.tool_manager.get_last_sources() == []


class TestRAGSystemToolRegistration:
    """Tests for tool registration in RAGSystem"""

    @pytest.fixture
    def rag_system(self, mock_config):
        """RAGSystem for testing tool registration"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults(
                documents=[], metadata=[], distances=[]
            )
            mock_vs_class.return_value = mock_vector_store

            return RAGSystem(mock_config)

    def test_rag_system_registers_search_tool(self, rag_system):
        """
        Given: RAGSystem initialized
        When: Checking registered tools
        Then: search_course_content tool is registered
        """
        tool_names = [t["name"] for t in rag_system.tool_manager.get_tool_definitions()]
        assert "search_course_content" in tool_names

    def test_rag_system_registers_outline_tool(self, rag_system):
        """
        Given: RAGSystem initialized
        When: Checking registered tools
        Then: get_course_outline tool is registered
        """
        tool_names = [t["name"] for t in rag_system.tool_manager.get_tool_definitions()]
        assert "get_course_outline" in tool_names

    def test_rag_system_has_exactly_two_tools(self, rag_system):
        """
        Given: RAGSystem initialized
        When: Checking tool count
        Then: Exactly two tools are registered
        """
        definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(definitions) == 2


class TestRAGSystemErrorHandling:
    """Tests for error handling in RAGSystem"""

    @pytest.fixture
    def rag_system(self, mock_config):
        """RAGSystem for testing error scenarios"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager') as mock_sm_class:

            mock_vector_store = Mock()
            mock_ai_generator = Mock()
            mock_session_manager = Mock()

            mock_vs_class.return_value = mock_vector_store
            mock_ai_class.return_value = mock_ai_generator
            mock_sm_class.return_value = mock_session_manager

            mock_session_manager.get_conversation_history.return_value = None
            mock_vector_store.search.return_value = SearchResults(
                documents=[], metadata=[], distances=[]
            )

            system = RAGSystem(mock_config)
            system._mock_ai = mock_ai_generator
            system._mock_session = mock_session_manager
            return system

    def test_query_handles_ai_generator_response(self, rag_system):
        """
        Given: AIGenerator returns a response
        When: query() is called
        Then: Response is returned correctly
        """
        rag_system._mock_ai.generate_response.return_value = "Test response"

        response, _ = rag_system.query("Test query")

        assert response == "Test response"
