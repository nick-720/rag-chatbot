"""
Tests for FastAPI endpoints - /api/query, /api/courses, /api/session/clear, /
"""
import pytest
from unittest.mock import Mock


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_200_with_valid_request(self, test_client):
        """
        Given: Valid query request
        When: POST to /api/query
        Then: Returns 200 status
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )
        assert response.status_code == 200

    def test_query_returns_answer_in_response(self, test_client):
        """
        Given: Valid query request
        When: POST to /api/query
        Then: Response contains answer field
        """
        response = test_client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer from RAG system"

    def test_query_returns_sources_list(self, test_client):
        """
        Given: Valid query request
        When: POST to /api/query
        Then: Response contains sources as list
        """
        response = test_client.post(
            "/api/query",
            json={"query": "Search query"}
        )
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Course A - Lesson 1"

    def test_query_creates_session_when_not_provided(self, test_client):
        """
        Given: Query without session_id
        When: POST to /api/query
        Then: Response includes generated session_id
        """
        response = test_client.post(
            "/api/query",
            json={"query": "No session query"}
        )
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_query_uses_provided_session_id(self, test_client, mock_rag_system):
        """
        Given: Query with existing session_id
        When: POST to /api/query
        Then: Uses provided session_id
        """
        response = test_client.post(
            "/api/query",
            json={"query": "Follow-up", "session_id": "my-session"}
        )
        data = response.json()
        assert data["session_id"] == "my-session"
        mock_rag_system.query.assert_called_with("Follow-up", "my-session")

    def test_query_calls_rag_system_with_query(self, test_client, mock_rag_system):
        """
        Given: Query request
        When: POST to /api/query
        Then: RAGSystem.query() is called with query text
        """
        test_client.post(
            "/api/query",
            json={"query": "Test question"}
        )
        mock_rag_system.query.assert_called()
        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == "Test question"

    def test_query_returns_422_for_missing_query(self, test_client):
        """
        Given: Request without query field
        When: POST to /api/query
        Then: Returns 422 validation error
        """
        response = test_client.post(
            "/api/query",
            json={}
        )
        assert response.status_code == 422

    def test_query_returns_422_for_empty_body(self, test_client):
        """
        Given: Empty request body
        When: POST to /api/query
        Then: Returns 422 validation error
        """
        response = test_client.post(
            "/api/query",
            content="",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_query_returns_500_when_rag_fails(self, test_client, mock_rag_system):
        """
        Given: RAGSystem raises exception
        When: POST to /api/query
        Then: Returns 500 with error detail
        """
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post(
            "/api/query",
            json={"query": "Failing query"}
        )
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_200(self, test_client):
        """
        Given: API is running
        When: GET /api/courses
        Then: Returns 200 status
        """
        response = test_client.get("/api/courses")
        assert response.status_code == 200

    def test_courses_returns_total_count(self, test_client):
        """
        Given: Courses are loaded
        When: GET /api/courses
        Then: Response includes total_courses count
        """
        response = test_client.get("/api/courses")
        data = response.json()
        assert "total_courses" in data
        assert data["total_courses"] == 2

    def test_courses_returns_course_titles(self, test_client):
        """
        Given: Courses are loaded
        When: GET /api/courses
        Then: Response includes course_titles list
        """
        response = test_client.get("/api/courses")
        data = response.json()
        assert "course_titles" in data
        assert data["course_titles"] == ["Course A", "Course B"]

    def test_courses_returns_500_on_error(self, test_client, mock_rag_system):
        """
        Given: Analytics retrieval fails
        When: GET /api/courses
        Then: Returns 500 error
        """
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")
        assert response.status_code == 500


@pytest.mark.api
class TestSessionClearEndpoint:
    """Tests for POST /api/session/clear endpoint"""

    def test_clear_session_returns_200(self, test_client):
        """
        Given: Valid session_id
        When: POST to /api/session/clear
        Then: Returns 200 status
        """
        response = test_client.post(
            "/api/session/clear",
            json={"query": "", "session_id": "session-to-clear"}
        )
        assert response.status_code == 200

    def test_clear_session_returns_confirmation(self, test_client):
        """
        Given: Valid session clear request
        When: POST to /api/session/clear
        Then: Returns confirmation message
        """
        response = test_client.post(
            "/api/session/clear",
            json={"query": "", "session_id": "my-session"}
        )
        data = response.json()
        assert data["message"] == "Session cleared"
        assert data["session_id"] == "my-session"

    def test_clear_session_calls_session_manager(self, test_client, mock_rag_system):
        """
        Given: Session clear request
        When: POST to /api/session/clear
        Then: SessionManager.clear_session() is called
        """
        test_client.post(
            "/api/session/clear",
            json={"query": "", "session_id": "target-session"}
        )
        mock_rag_system.session_manager.clear_session.assert_called_once_with("target-session")

    def test_clear_session_handles_missing_session_id(self, test_client, mock_rag_system):
        """
        Given: Request without session_id
        When: POST to /api/session/clear
        Then: Does not call clear_session, returns gracefully
        """
        response = test_client.post(
            "/api/session/clear",
            json={"query": ""}
        )
        assert response.status_code == 200
        mock_rag_system.session_manager.clear_session.assert_not_called()


@pytest.mark.api
class TestRootEndpoint:
    """Tests for GET / health check endpoint"""

    def test_root_returns_200(self, test_client):
        """
        Given: API is running
        When: GET /
        Then: Returns 200 status
        """
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_status_ok(self, test_client):
        """
        Given: API is running
        When: GET /
        Then: Returns status ok
        """
        response = test_client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_service_name(self, test_client):
        """
        Given: API is running
        When: GET /
        Then: Returns service identifier
        """
        response = test_client.get("/")
        data = response.json()
        assert "service" in data


@pytest.mark.api
class TestResponseSchema:
    """Tests for API response schema validation"""

    def test_query_response_matches_schema(self, test_client):
        """
        Given: Valid query
        When: POST to /api/query
        Then: Response matches QueryResponse schema
        """
        response = test_client.post(
            "/api/query",
            json={"query": "Test"}
        )
        data = response.json()

        # Required fields
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_source_includes_text_and_url(self, test_client):
        """
        Given: Query that returns sources
        When: POST to /api/query
        Then: Each source has text and optional url
        """
        response = test_client.post(
            "/api/query",
            json={"query": "With sources"}
        )
        data = response.json()
        source = data["sources"][0]

        assert "text" in source
        assert "url" in source
        assert source["url"] == "https://example.com/lesson1"

    def test_courses_response_matches_schema(self, test_client):
        """
        Given: API is running
        When: GET /api/courses
        Then: Response matches CourseStats schema
        """
        response = test_client.get("/api/courses")
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
