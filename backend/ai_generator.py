import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may use up to 2 tool calls per query if needed to fully answer the question
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Tool Usage:
- Use the outline tool when users ask about course structure, outline, table of contents, or lesson lists
- Returns course title, course link, and complete lesson list with lesson numbers and titles
- Use this instead of search when the user wants to know what lessons a course contains or asks for an overview

Sequential Tool Use:
- For complex queries, you may use one tool, review the results, then use another tool
- Example: Get a course outline first, then search for specific content based on what you learned
- Each tool call helps build context for a complete answer

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling up to max_rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool call rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution recursively if needed
        if response.stop_reason == "tool_use" and tool_manager and tools:
            return self._process_tool_chain(
                response=response,
                messages=api_params["messages"].copy(),
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
                current_round=1,
                max_rounds=max_rounds
            )

        # Return direct response
        return response.content[0].text
    
    def _process_tool_chain(self, response, messages: List[Dict[str, Any]],
                            system_content: str, tools: List, tool_manager,
                            current_round: int, max_rounds: int) -> str:
        """
        Recursively process tool calls, allowing up to max_rounds of tool execution.

        Args:
            response: The Claude response containing tool use requests
            messages: Accumulated message history
            system_content: System prompt content
            tools: Tool definitions for API calls
            tool_manager: Manager to execute tools
            current_round: Current round number (1-indexed)
            max_rounds: Maximum number of tool rounds allowed

        Returns:
            Final response text after all tool processing
        """
        # Append assistant's tool_use response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        # Append tool results to messages
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Determine if we should include tools in next call
        include_tools = current_round < max_rounds

        # Build next API call params
        next_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        if include_tools:
            next_params["tools"] = tools
            next_params["tool_choice"] = {"type": "auto"}

        # Make next API call
        next_response = self.client.messages.create(**next_params)

        # Recursive case: More tool calls requested and rounds remaining
        if (next_response.stop_reason == "tool_use"
            and include_tools
            and current_round < max_rounds):
            return self._process_tool_chain(
                response=next_response,
                messages=messages,
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
                current_round=current_round + 1,
                max_rounds=max_rounds
            )

        # Base case: Return final text response
        return next_response.content[0].text