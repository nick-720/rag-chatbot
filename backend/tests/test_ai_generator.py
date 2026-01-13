"""
Tests for AIGenerator class - Claude API integration and tool calling.
"""

import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


class TestAIGeneratorGenerateResponse:
    """Tests for AIGenerator.generate_response() method"""

    @pytest.fixture
    def ai_generator(self):
        """AIGenerator with mocked Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            generator = AIGenerator(api_key="test-key", model="claude-test")
            generator.client = mock_client
            return generator

    # ===== Test Case 1: Direct response (no tool use) =====
    def test_generate_response_returns_text_when_no_tool_use(
        self, ai_generator, mock_text_response
    ):
        """
        Given: Claude returns a direct text response
        When: generate_response() is called
        Then: Returns the text content directly
        """
        ai_generator.client.messages.create.return_value = mock_text_response

        result = ai_generator.generate_response(query="What is 2+2?")

        assert result == "This is a direct answer."
        ai_generator.client.messages.create.assert_called_once()

    def test_generate_response_passes_query_in_messages(
        self, ai_generator, mock_text_response
    ):
        """
        Given: A query string
        When: generate_response() is called
        Then: Query is passed in messages array as user role
        """
        ai_generator.client.messages.create.return_value = mock_text_response

        ai_generator.generate_response(query="Test question")

        call_args = ai_generator.client.messages.create.call_args
        messages = call_args.kwargs.get("messages")
        assert messages[0]["role"] == "user"
        assert "Test question" in messages[0]["content"]

    # ===== Test Case 2: Response with tool_use block =====
    def test_generate_response_detects_tool_use_stop_reason(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Claude returns tool_use stop_reason
        When: generate_response() is called with tool_manager
        Then: Executes tool and makes second API call
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"

        result = ai_generator.generate_response(
            query="Search for MCP",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert ai_generator.client.messages.create.call_count == 2
        assert "Based on the course content" in result

    def test_generate_response_calls_tool_manager_with_correct_params(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Claude requests search_course_content tool
        When: generate_response() handles tool use
        Then: Calls tool_manager.execute_tool with correct name and params
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "search results"

        ai_generator.generate_response(
            query="Search query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP basics", course_name="MCP"
        )

    # ===== Test Case 3: Multiple tool calls in one response =====
    def test_generate_response_handles_multiple_tool_calls(
        self, ai_generator, mock_multi_tool_response, mock_final_response
    ):
        """
        Given: Claude returns multiple tool_use blocks
        When: generate_response() is called
        Then: All tools are executed
        """
        ai_generator.client.messages.create.side_effect = [
            mock_multi_tool_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        ai_generator.generate_response(
            query="Complex query",
            tools=[{"name": "search"}, {"name": "outline"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_tool_manager.execute_tool.call_count == 2

    # ===== Test Case 4: Tool execution and second API call =====
    def test_generate_response_final_call_excludes_tools_after_max_rounds(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: First API call returns tool_use and max_rounds=1
        When: Tool is executed and second call made
        Then: Second call does NOT include tools parameter (max rounds reached)
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        ai_generator.generate_response(
            query="Search query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=1,
        )

        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        assert "tools" not in second_call_args.kwargs

    def test_generate_response_second_call_includes_tool_results(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Tool execution returns results
        When: Second API call is made
        Then: Messages include tool_result content
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "The search found X, Y, Z"

        ai_generator.generate_response(
            query="Search query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
        )

        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs.get("messages")

        # Should have: original user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        tool_results = messages[2]["content"]
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["content"] == "The search found X, Y, Z"

    def test_generate_response_preserves_tool_use_id_in_result(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Tool use has specific ID
        When: Tool result is sent back
        Then: tool_use_id matches the original tool use
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "result"

        ai_generator.generate_response(
            query="Query", tools=[{"name": "search"}], tool_manager=mock_tool_manager
        )

        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs.get("messages")
        tool_result = messages[2]["content"][0]
        assert tool_result["tool_use_id"] == "tool_123"

    # ===== Test Case 5: Conversation history handling =====
    def test_generate_response_includes_history_in_system(
        self, ai_generator, mock_text_response
    ):
        """
        Given: Conversation history is provided
        When: generate_response() is called
        Then: History is appended to system prompt
        """
        ai_generator.client.messages.create.return_value = mock_text_response
        history = "User: Previous question\nAssistant: Previous answer"

        ai_generator.generate_response(query="Follow-up", conversation_history=history)

        call_args = ai_generator.client.messages.create.call_args
        system = call_args.kwargs.get("system")
        assert "Previous conversation:" in system
        assert "Previous question" in system

    def test_generate_response_without_history_uses_base_system_prompt(
        self, ai_generator, mock_text_response
    ):
        """
        Given: No conversation history provided
        When: generate_response() is called
        Then: Uses only the base system prompt
        """
        ai_generator.client.messages.create.return_value = mock_text_response

        ai_generator.generate_response(query="Simple question")

        call_args = ai_generator.client.messages.create.call_args
        system = call_args.kwargs.get("system")
        assert "Previous conversation:" not in system
        assert "AI assistant specialized in course materials" in system

    def test_generate_response_passes_tools_when_provided(
        self, ai_generator, mock_text_response
    ):
        """
        Given: Tools list is provided
        When: generate_response() is called
        Then: Tools are included in API call with auto choice
        """
        ai_generator.client.messages.create.return_value = mock_text_response
        tools = [{"name": "search_course_content", "description": "Search"}]

        ai_generator.generate_response(query="Question", tools=tools)

        call_args = ai_generator.client.messages.create.call_args
        assert call_args.kwargs.get("tools") == tools
        assert call_args.kwargs.get("tool_choice") == {"type": "auto"}

    def test_generate_response_without_tools_omits_tool_params(
        self, ai_generator, mock_text_response
    ):
        """
        Given: No tools provided
        When: generate_response() is called
        Then: Tools parameters are not in API call
        """
        ai_generator.client.messages.create.return_value = mock_text_response

        ai_generator.generate_response(query="Question")

        call_args = ai_generator.client.messages.create.call_args
        assert "tools" not in call_args.kwargs
        assert "tool_choice" not in call_args.kwargs


class TestAIGeneratorProcessToolChain:
    """Tests for AIGenerator._process_tool_chain() method"""

    @pytest.fixture
    def ai_generator_with_mock_client(self):
        """AIGenerator with directly mocked client"""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test", model="test")
            generator.client = Mock()
            return generator

    def test_process_tool_chain_returns_final_text(
        self, ai_generator_with_mock_client, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Initial response with tool_use
        When: _process_tool_chain() is called
        Then: Returns text from final response
        """
        generator = ai_generator_with_mock_client
        generator.client.messages.create.return_value = mock_final_response
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "tool output"

        messages = [{"role": "user", "content": "query"}]
        tools = [{"name": "search"}]

        result = generator._process_tool_chain(
            response=mock_tool_use_response,
            messages=messages,
            system_content="system prompt",
            tools=tools,
            tool_manager=mock_tool_manager,
            current_round=1,
            max_rounds=2,
        )

        assert result == "Based on the course content, here is your answer."

    def test_process_tool_chain_preserves_original_messages(
        self, ai_generator_with_mock_client, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Messages list with user query
        When: _process_tool_chain() runs
        Then: Original user message is preserved in API call
        """
        generator = ai_generator_with_mock_client
        generator.client.messages.create.return_value = mock_final_response
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "result"

        original_message = {"role": "user", "content": "original query"}
        messages = [original_message]
        tools = [{"name": "search"}]

        generator._process_tool_chain(
            response=mock_tool_use_response,
            messages=messages,
            system_content="test system",
            tools=tools,
            tool_manager=mock_tool_manager,
            current_round=1,
            max_rounds=2,
        )

        call_args = generator.client.messages.create.call_args
        final_messages = call_args.kwargs.get("messages")
        assert final_messages[0] == original_message

    def test_process_tool_chain_adds_assistant_response(
        self, ai_generator_with_mock_client, mock_tool_use_response, mock_final_response
    ):
        """
        Given: Initial response with tool_use content
        When: _process_tool_chain() runs
        Then: Assistant's tool_use response is added to messages
        """
        generator = ai_generator_with_mock_client
        generator.client.messages.create.return_value = mock_final_response
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "result"

        messages = [{"role": "user", "content": "query"}]
        tools = [{"name": "search"}]

        generator._process_tool_chain(
            response=mock_tool_use_response,
            messages=messages,
            system_content="system",
            tools=tools,
            tool_manager=mock_tool_manager,
            current_round=1,
            max_rounds=2,
        )

        call_args = generator.client.messages.create.call_args
        final_messages = call_args.kwargs.get("messages")
        assert final_messages[1]["role"] == "assistant"
        assert final_messages[1]["content"] == mock_tool_use_response.content


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator initialization and configuration"""

    def test_init_sets_model_in_base_params(self):
        """
        Given: Model name provided to constructor
        When: AIGenerator is initialized
        Then: Model is set in base_params
        """
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test", model="claude-3-opus")

        assert generator.base_params["model"] == "claude-3-opus"

    def test_init_sets_default_temperature_and_max_tokens(self):
        """
        Given: AIGenerator initialized
        When: Checking base_params
        Then: Has temperature=0 and max_tokens=800
        """
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test", model="test")

        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_init_creates_anthropic_client(self):
        """
        Given: API key provided
        When: AIGenerator is initialized
        Then: Anthropic client is created with the key
        """
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            AIGenerator(api_key="my-api-key", model="test")

        mock_anthropic.assert_called_once_with(api_key="my-api-key")

    def test_system_prompt_contains_key_instructions(self):
        """
        Given: AIGenerator class
        When: Checking SYSTEM_PROMPT
        Then: Contains key instructions for behavior
        """
        assert "course materials" in AIGenerator.SYSTEM_PROMPT
        assert (
            "search tool" in AIGenerator.SYSTEM_PROMPT.lower()
            or "Search Tool" in AIGenerator.SYSTEM_PROMPT
        )


class TestSequentialToolCalling:
    """Tests for sequential tool calling (up to 2 rounds)"""

    @pytest.fixture
    def ai_generator(self):
        """AIGenerator with mocked Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            generator = AIGenerator(api_key="test-key", model="claude-test")
            generator.client = mock_client
            return generator

    def test_two_sequential_tool_calls_makes_three_api_calls(
        self,
        ai_generator,
        mock_tool_use_response,
        mock_second_tool_use_response,
        mock_final_response,
    ):
        """
        Given: Claude requests tool in round 1, then another tool in round 2
        When: generate_response() is called with max_rounds=2
        Then: Makes 3 API calls total (initial + 2 tool rounds)
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_second_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        result = ai_generator.generate_response(
            query="Complex query needing multiple searches",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        assert ai_generator.client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Based on the course content, here is your answer."

    def test_second_round_includes_tools_when_rounds_remaining(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: First API call returns tool_use with max_rounds=2
        When: Second API call is made (round 1 < max_rounds)
        Then: Tools are included in second call
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        ai_generator.generate_response(
            query="Query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        assert "tools" in second_call_args.kwargs
        assert second_call_args.kwargs["tools"] == [{"name": "search"}]

    def test_third_call_excludes_tools_after_two_rounds(
        self,
        ai_generator,
        mock_tool_use_response,
        mock_second_tool_use_response,
        mock_final_response,
    ):
        """
        Given: Claude requests tools in both round 1 and round 2
        When: Third API call is made (after max_rounds=2)
        Then: Tools are NOT included (forces text response)
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_second_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        ai_generator.generate_response(
            query="Complex query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        third_call_args = ai_generator.client.messages.create.call_args_list[2]
        assert "tools" not in third_call_args.kwargs

    def test_early_termination_when_no_tool_use_in_response(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: First call returns tool_use, second returns text
        When: generate_response() is called
        Then: Only 2 API calls made (terminates when no tool_use)
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        result = ai_generator.generate_response(
            query="Simple query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        assert ai_generator.client.messages.create.call_count == 2
        assert result == "Based on the course content, here is your answer."

    def test_messages_accumulate_across_rounds(
        self,
        ai_generator,
        mock_tool_use_response,
        mock_second_tool_use_response,
        mock_final_response,
    ):
        """
        Given: Two rounds of tool calls
        When: Final API call is made
        Then: Messages array contains full conversation history
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_second_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        ai_generator.generate_response(
            query="Multi-step query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        third_call_args = ai_generator.client.messages.create.call_args_list[2]
        messages = third_call_args.kwargs.get("messages")

        # Should have: user query, assistant tool_use 1, user tool_result 1,
        #              assistant tool_use 2, user tool_result 2
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_max_rounds_one_behaves_like_single_round(
        self, ai_generator, mock_tool_use_response, mock_final_response
    ):
        """
        Given: max_rounds=1
        When: First call returns tool_use
        Then: Second call does not include tools (single round behavior)
        """
        ai_generator.client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response,
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        ai_generator.generate_response(
            query="Query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager,
            max_rounds=1,
        )

        assert ai_generator.client.messages.create.call_count == 2
        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        assert "tools" not in second_call_args.kwargs
