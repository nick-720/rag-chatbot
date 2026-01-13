"""
Shared fixtures and mocks for RAG chatbot tests.
"""
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from session_manager import SessionManager


# ============ API Test App & Client ============

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    """Model for a source citation"""
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API tests"""
    rag = Mock()
    rag.query.return_value = ("Test answer from RAG system", [
        {"text": "Course A - Lesson 1", "url": "https://example.com/lesson1"}
    ])
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test-session-123"
    rag.session_manager.clear_session.return_value = None
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"]
    }
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app with endpoints defined inline.
    This avoids importing the main app which mounts static files.
    """
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: QueryRequest):
        """Clear conversation history for a session."""
        if request.session_id:
            mock_rag_system.session_manager.clear_session(request.session_id)
        return {"message": "Session cleared", "session_id": request.session_id}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {"status": "ok", "service": "RAG Chatbot API"}

    return app


@pytest.fixture
def test_client(test_app):
    """TestClient for making requests to the test app"""
    return TestClient(test_app)


# ============ SearchResults Factories ============

@pytest.fixture
def sample_search_results():
    """Factory fixture for creating SearchResults with sample data"""
    def _create(
        documents: List[str] = None,
        metadata: List[Dict] = None,
        distances: List[float] = None,
        error: str = None
    ):
        return SearchResults(
            documents=documents if documents is not None else ["Sample content about MCP protocols"],
            metadata=metadata if metadata is not None else [{"course_title": "MCP Course", "lesson_number": 1}],
            distances=distances if distances is not None else [0.15],
            error=error
        )
    return _create


@pytest.fixture
def empty_search_results():
    """Empty search results fixture"""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """Search results with error"""
    return SearchResults.empty("No course found matching 'NonExistent'")


# ============ Mock VectorStore ============

@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with configurable behavior"""
    store = Mock()

    # Default successful search behavior
    store.search.return_value = SearchResults(
        documents=["Content about AI basics", "More AI content"],
        metadata=[
            {"course_title": "AI Fundamentals", "lesson_number": 1},
            {"course_title": "AI Fundamentals", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )

    # Default lesson link behavior
    store.get_lesson_link.return_value = "https://example.com/lesson/1"

    # Default course resolution
    store._resolve_course_name.return_value = "AI Fundamentals"

    # Course catalog mock for outline tool
    store.course_catalog = Mock()
    store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'AI Fundamentals',
            'course_link': 'https://example.com/ai-course',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson/1"}]'
        }]
    }

    return store


# ============ Claude API Response Mocks ============

@dataclass
class MockContentBlock:
    """Mock content block for Claude responses"""
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: Dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockClaudeResponse:
    """Mock Claude API response"""
    content: List[MockContentBlock]
    stop_reason: str


@pytest.fixture
def mock_text_response():
    """Mock Claude response with direct text (no tool use)"""
    return MockClaudeResponse(
        content=[MockContentBlock(type="text", text="This is a direct answer.")],
        stop_reason="end_turn"
    )


@pytest.fixture
def mock_tool_use_response():
    """Mock Claude response requesting tool use"""
    return MockClaudeResponse(
        content=[
            MockContentBlock(
                type="tool_use",
                id="tool_123",
                name="search_course_content",
                input={"query": "MCP basics", "course_name": "MCP"}
            )
        ],
        stop_reason="tool_use"
    )


@pytest.fixture
def mock_multi_tool_response():
    """Mock Claude response with multiple tool calls"""
    return MockClaudeResponse(
        content=[
            MockContentBlock(
                type="tool_use",
                id="tool_1",
                name="search_course_content",
                input={"query": "first topic"}
            ),
            MockContentBlock(
                type="tool_use",
                id="tool_2",
                name="get_course_outline",
                input={"course_name": "MCP Course"}
            )
        ],
        stop_reason="tool_use"
    )


@pytest.fixture
def mock_final_response():
    """Mock Claude final response after tool execution"""
    return MockClaudeResponse(
        content=[MockContentBlock(type="text", text="Based on the course content, here is your answer.")],
        stop_reason="end_turn"
    )


@pytest.fixture
def mock_second_tool_use_response():
    """Mock Claude response requesting a second tool after first tool result"""
    return MockClaudeResponse(
        content=[
            MockContentBlock(
                type="tool_use",
                id="tool_456",
                name="search_course_content",
                input={"query": "advanced topic", "course_name": "Advanced Course"}
            )
        ],
        stop_reason="tool_use"
    )


# ============ Mock Anthropic Client ============

@pytest.fixture
def mock_anthropic_client(mock_text_response):
    """Mock Anthropic client with configurable responses"""
    client = Mock()
    client.messages.create.return_value = mock_text_response
    return client


# ============ Tool Manager Fixtures ============

@pytest.fixture
def tool_manager_with_search(mock_vector_store):
    """ToolManager with CourseSearchTool registered"""
    manager = ToolManager()
    search_tool = CourseSearchTool(mock_vector_store)
    manager.register_tool(search_tool)
    return manager


@pytest.fixture
def tool_manager_full(mock_vector_store):
    """ToolManager with all tools registered"""
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))
    manager.register_tool(CourseOutlineTool(mock_vector_store))
    return manager


# ============ Session Manager Fixture ============

@pytest.fixture
def session_manager():
    """Fresh SessionManager instance"""
    return SessionManager(max_history=2)


# ============ Mock Config ============

@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.EMBEDDING_MODEL = "test-model"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "test-model"
    config.MAX_HISTORY = 2
    return config
