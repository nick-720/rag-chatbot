# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials. It uses semantic search with ChromaDB and Anthropic's Claude API to answer questions about educational content.

## Commands

**Run the application:**
```bash
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Important:**
- Always use `uv run` to execute Python commands. Do not use `pip` directly.
- Always use `uv` to run Python files
- Use `uv add <package>` to add dependencies (updates `pyproject.toml` and `uv.lock`). 

The web interface runs at `http://localhost:8000`, API docs at `http://localhost:8000/docs`.

## Architecture

### Query Flow
1. Frontend (`frontend/script.js`) sends POST to `/api/query` with `{query, session_id}`
2. FastAPI (`backend/app.py`) delegates to `RAGSystem.query()`
3. `RAGSystem` (`backend/rag_system.py`) orchestrates the components:
   - Fetches conversation history from `SessionManager`
   - Calls `AIGenerator.generate_response()` with tool definitions
4. `AIGenerator` (`backend/ai_generator.py`) makes Claude API call with tools enabled
5. Claude decides to use `search_course_content` tool → `ToolManager` executes it
6. `CourseSearchTool` (`backend/search_tools.py`) queries `VectorStore` (ChromaDB)
7. Second Claude API call synthesizes search results into natural language
8. Response returns with answer and sources list

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| RAGSystem | `backend/rag_system.py` | Main orchestrator |
| AIGenerator | `backend/ai_generator.py` | Claude API interactions, tool execution loop |
| VectorStore | `backend/vector_store.py` | ChromaDB wrapper, semantic search |
| DocumentProcessor | `backend/document_processor.py` | Parses course files, chunks text |
| SessionManager | `backend/session_manager.py` | Conversation history per session |
| ToolManager + CourseSearchTool | `backend/search_tools.py` | Tool definitions and execution |

### Document Format
Course files in `docs/` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[content...]
```

Documents are chunked (~800 chars with 100 char overlap) and stored in ChromaDB on startup.

### Tool Calling Pattern
This uses **native Anthropic tool calling**, not MCP. Tools are defined in `search_tools.py` and passed directly to the Claude API. The `AIGenerator` handles the two-call pattern: first call returns `tool_use`, tool is executed locally, second call synthesizes the answer.

## Configuration

Key settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100
- `MAX_RESULTS`: 5 (search results per query)
- `MAX_HISTORY`: 2 (conversation exchanges kept)

Requires `ANTHROPIC_API_KEY` in `.env` file.
