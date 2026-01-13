"""
Tests for CourseSearchTool, CourseOutlineTool, and ToolManager.
"""

import pytest
from unittest.mock import Mock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    # ===== Test Case 1: Successful search with results =====
    def test_execute_successful_search_returns_formatted_results(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: VectorStore returns valid search results
        When: execute() is called with a query
        Then: Returns formatted string with course headers and content
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="What is MCP?")

        assert "[MCP Course - Lesson 1]" in result
        assert "Sample content about MCP protocols" in result
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

    # ===== Test Case 2: Search with no results =====
    def test_execute_no_results_returns_not_found_message(
        self, mock_vector_store, empty_search_results
    ):
        """
        Given: VectorStore returns empty results
        When: execute() is called
        Then: Returns appropriate "no content found" message
        """
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    # ===== Test Case 3: Search with course_name filter =====
    def test_execute_with_course_filter_passes_to_vector_store(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: User specifies a course_name filter
        When: execute() is called with course_name
        Then: Passes course_name to VectorStore.search()
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="protocols", course_name="MCP Course")

        mock_vector_store.search.assert_called_once_with(
            query="protocols", course_name="MCP Course", lesson_number=None
        )

    def test_execute_with_course_filter_includes_course_in_not_found_message(
        self, mock_vector_store, empty_search_results
    ):
        """
        Given: Course filter specified but no results found
        When: execute() returns not found message
        Then: Message includes the course name context
        """
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="something", course_name="MCP")

        assert "in course 'MCP'" in result

    # ===== Test Case 4: Search with lesson_number filter =====
    def test_execute_with_lesson_filter_passes_to_vector_store(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: User specifies a lesson_number filter
        When: execute() is called with lesson_number
        Then: Passes lesson_number to VectorStore.search()
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="introduction", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_lesson_filter_includes_lesson_in_not_found_message(
        self, mock_vector_store, empty_search_results
    ):
        """
        Given: Lesson filter specified but no results found
        When: execute() returns not found message
        Then: Message includes the lesson number context
        """
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="something", lesson_number=3)

        assert "in lesson 3" in result

    # ===== Test Case 5: Search with both filters =====
    def test_execute_with_both_filters_passes_both_to_vector_store(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: User specifies both course_name and lesson_number
        When: execute() is called with both filters
        Then: Both are passed to VectorStore.search()
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="setup", course_name="MCP", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="setup", course_name="MCP", lesson_number=2
        )

    def test_execute_with_both_filters_not_found_includes_both_in_message(
        self, mock_vector_store, empty_search_results
    ):
        """
        Given: Both filters specified but no results
        When: execute() returns not found message
        Then: Message includes both course and lesson context
        """
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="something", course_name="MCP", lesson_number=3)

        assert "in course 'MCP'" in result
        assert "in lesson 3" in result

    # ===== Test Case 6: VectorStore returns error =====
    def test_execute_returns_error_when_vector_store_fails(self, mock_vector_store):
        """
        Given: VectorStore.search() returns an error
        When: execute() is called
        Then: Returns the error message directly
        """
        error_results = SearchResults.empty("No course found matching 'FakeCourse'")
        mock_vector_store.search.return_value = error_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything", course_name="FakeCourse")

        assert "No course found matching 'FakeCourse'" in result

    # ===== Test Case 7: Source tracking correctness =====
    def test_execute_tracks_sources_with_urls(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: Search returns results with metadata
        When: execute() completes
        Then: last_sources contains correct source dicts with text and URL
        """
        mock_vector_store.search.return_value = sample_search_results()
        mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/mcp/lesson1"
        )
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="MCP basics")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "MCP Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/mcp/lesson1"

    def test_execute_tracks_sources_without_lesson_number(self, mock_vector_store):
        """
        Given: Search results include item without lesson_number
        When: execute() completes
        Then: Source text excludes lesson suffix
        """
        results = SearchResults(
            documents=["Course intro content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": None}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="introduction")

        assert tool.last_sources[0]["text"] == "MCP Course"

    def test_execute_tracks_multiple_sources(self, mock_vector_store):
        """
        Given: Search returns multiple results
        When: execute() completes
        Then: last_sources contains all sources in order
        """
        results = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
                {"course_title": "Course B", "lesson_number": 1},
            ],
            distances=[0.1, 0.2, 0.3],
        )
        mock_vector_store.search.return_value = results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="topic")

        assert len(tool.last_sources) == 3
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[2]["text"] == "Course B - Lesson 1"

    def test_execute_clears_previous_sources_on_new_search(
        self, mock_vector_store, sample_search_results
    ):
        """
        Given: Tool has sources from previous search
        When: New search is executed
        Then: last_sources is replaced with new sources
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = [{"text": "old source", "url": "old-url"}]

        tool.execute(query="new query")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] != "old source"


class TestCourseSearchToolFormatResults:
    """Tests for CourseSearchTool._format_results() method"""

    def test_format_results_includes_course_and_lesson_header(self, mock_vector_store):
        """
        Given: SearchResults with course_title and lesson_number
        When: _format_results() is called
        Then: Output includes formatted header [Course - Lesson N]
        """
        tool = CourseSearchTool(mock_vector_store)
        results = SearchResults(
            documents=["Some content here"],
            metadata=[{"course_title": "AI Course", "lesson_number": 5}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[AI Course - Lesson 5]" in formatted
        assert "Some content here" in formatted

    def test_format_results_handles_missing_lesson_number(self, mock_vector_store):
        """
        Given: SearchResults without lesson_number in metadata
        When: _format_results() is called
        Then: Header shows only course title without lesson suffix
        """
        tool = CourseSearchTool(mock_vector_store)
        results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "General Course"}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[General Course]" in formatted
        assert "Lesson" not in formatted

    def test_format_results_separates_multiple_results(self, mock_vector_store):
        """
        Given: SearchResults with multiple documents
        When: _format_results() is called
        Then: Results are separated by double newlines
        """
        tool = CourseSearchTool(mock_vector_store)
        results = SearchResults(
            documents=["First doc", "Second doc"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        formatted = tool._format_results(results)

        assert "\n\n" in formatted
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()"""

    def test_get_tool_definition_returns_valid_anthropic_format(
        self, mock_vector_store
    ):
        """
        Given: CourseSearchTool instance
        When: get_tool_definition() is called
        Then: Returns dict matching Anthropic tool schema
        """
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool"""

    def test_execute_returns_formatted_outline(self, mock_vector_store):
        """
        Given: Valid course name
        When: execute() is called
        Then: Returns formatted outline with title, link, lessons
        """
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="AI")

        assert "Course: AI Fundamentals" in result
        assert "Link:" in result
        assert "Lessons:" in result

    def test_execute_returns_not_found_for_invalid_course(self, mock_vector_store):
        """
        Given: Course name that doesn't resolve
        When: execute() is called
        Then: Returns appropriate not found message
        """
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="NonExistent")

        assert "No course found matching 'NonExistent'" in result

    def test_get_tool_definition_returns_valid_format(self, mock_vector_store):
        """
        Given: CourseOutlineTool instance
        When: get_tool_definition() is called
        Then: Returns valid Anthropic tool definition
        """
        tool = CourseOutlineTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["required"]


class TestToolManager:
    """Tests for ToolManager class"""

    def test_register_tool_adds_to_tools_dict(self, mock_vector_store):
        """
        Given: Empty ToolManager
        When: Tool is registered
        Then: Tool is accessible by name
        """
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions_returns_all_registered(self, tool_manager_full):
        """
        Given: ToolManager with multiple tools registered
        When: get_tool_definitions() is called
        Then: Returns list of all tool definitions
        """
        definitions = tool_manager_full.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_calls_correct_tool(
        self, tool_manager_with_search, mock_vector_store, sample_search_results
    ):
        """
        Given: ToolManager with registered tools
        When: execute_tool() is called with tool name
        Then: Correct tool's execute() is called with kwargs
        """
        mock_vector_store.search.return_value = sample_search_results()

        result = tool_manager_with_search.execute_tool(
            "search_course_content", query="test query", course_name="MCP"
        )

        mock_vector_store.search.assert_called_once()
        assert isinstance(result, str)

    def test_execute_tool_returns_not_found_for_unknown_tool(self):
        """
        Given: ToolManager
        When: execute_tool() called with unregistered tool name
        Then: Returns "Tool not found" message
        """
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources_returns_from_search_tool(
        self, tool_manager_with_search, mock_vector_store, sample_search_results
    ):
        """
        Given: Search tool has executed and has sources
        When: get_last_sources() is called
        Then: Returns the sources from the search tool
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool_manager_with_search.execute_tool("search_course_content", query="test")

        sources = tool_manager_with_search.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources_clears_all_tool_sources(
        self, tool_manager_with_search, mock_vector_store, sample_search_results
    ):
        """
        Given: Tools have accumulated sources
        When: reset_sources() is called
        Then: All tools' last_sources are cleared
        """
        mock_vector_store.search.return_value = sample_search_results()
        tool_manager_with_search.execute_tool("search_course_content", query="test")

        tool_manager_with_search.reset_sources()

        assert tool_manager_with_search.get_last_sources() == []

    def test_register_tool_raises_error_without_name(self):
        """
        Given: A tool with invalid definition (no name)
        When: register_tool() is called
        Then: Raises ValueError
        """
        manager = ToolManager()
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="must have a 'name'"):
            manager.register_tool(bad_tool)
