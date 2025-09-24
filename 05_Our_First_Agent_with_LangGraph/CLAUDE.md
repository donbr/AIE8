# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based agent project that implements agentic RAG (Retrieval-Augmented Generation) using LangChain, LangGraph, and LangSmith. The project demonstrates building simple search agents with tool capabilities and evaluation frameworks.

## Key Dependencies and Setup

This project uses `uv` for Python dependency management. The project requires Python 3.13 (specified in `.python-version`) and the following API keys need to be set:
- `OPENAI_API_KEY` - For LLM access (uses gpt-4.1-nano and gpt-4.1-mini models)
- `TAVILY_API_KEY` - For web search functionality
- `LANGCHAIN_API_KEY` - For LangSmith tracing and evaluation
- `LANGCHAIN_TRACING_V2="true"` - Enables tracing
- `LANGCHAIN_PROJECT` - Automatically set with format `AIE8 - LangGraph - {uuid}`

## Development Commands

### Dependency Management
```bash
# Install dependencies using uv
uv sync

# Add a new dependency
uv add <package_name>

# Activate the virtual environment
source .venv/bin/activate
```

### Running Code

- launch langgraph-langsmith.ipynb notebook to see the implementation and evaluation in action.

## Architecture

### Core Components

1. **Agent State Management**: Uses `TypedDict` with `add_messages` annotation to track conversation state through the graph.

2. **Graph Implementations**:
   - **Simple agent graph** (`simple_agent_graph`): Routes between "agent" and "action" nodes based on tool calls
   - **Helpfulness-checking graph** (`agent_with_helpfulness_check`): Adds LLM-based quality assessment that can retry responses up to 10 times

3. **Tool Integration**:
   - Tavily Search for web search
   - ArXiv Query for academic paper search
   - Tools are bound to the LLM model and executed through `ToolNode`

4. **Evaluation Framework**:
   - LangSmith integration for experiment tracking with automatic dataset creation
     - `correctness_evaluator`: LLM-as-judge using OpenAI o3-mini model
   - Custom evaluators:
     - `must_mention`: Checks if required phrases appear in output
   - Evaluation runs with `client.evaluate()` supporting concurrent execution

### Key Design Patterns

- **Conditional routing**:
  - `should_continue`: Routes to "action" node if tool calls present, otherwise ends
  - `tool_call_or_helpful`: Extends routing with helpfulness evaluation using GPT-4.1-mini
- **Tool binding**: Models are bound with tools using `bind_tools()` for seamless function calling
- **Async streaming**: Uses `astream()` with `stream_mode="updates"` for real-time node updates
- **Input/Output formatting**: `convert_inputs` and `parse_output` functions wrap the graph for LangSmith evaluation compatibility

## Important Files

- `langgraph-langsmith.ipynb`: Main implementation demonstrating both simple and advanced agent architectures with evaluation