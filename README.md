# MTG Card Search Agent

A multi-agent system for finding Magic: The Gathering cards using natural language queries. Built with PydanticAI, powered by OpenAI's GPT models, and integrated with the Scryfall API for comprehensive MTG card data.

<div align="center">
  <img src="assets/mtg-search.gif" alt="MTG Search Demo" />
</div>

## Features

- **Natural Language Processing**: Convert plain English requests into Scryfall search syntax
- **Multi-Agent Architecture**: Query generation and evaluation agents work together
- **Tag Similarity Search**: Fuzzy matching against comprehensive MTG tag database
- **Intelligent Iteration**: Up to 5 search refinement loops with agent feedback
- **Relevance Scoring**: 1-10 scoring system for card relevance
- **Installable Package**: Clean Python package structure for easy integration

## Installation

### Option 1: Install from Source (Development)

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd SearchAgent
    ```

2. **Create a virtual environment and install in development mode**:
    - Windows (PowerShell):
       ```powershell
       py -m venv .venv
       .venv\Scripts\Activate.ps1
       pip install -e .
       ```
    - macOS/Linux:
       ```bash
       python3 -m venv .venv
       source .venv/bin/activate
       pip install -e .
       ```

3. **Set up OpenAI API Key**:
    Create a `.env` file in the project root with:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```

### Option 2: Install as Package

1. **Install from wheel** (once built):
   ```bash
   pip install mtg-search-agent
   ```

2. **Set up OpenAI API Key**:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### CLI Interface (Recommended)

**After installation via pip:**
```bash
# Interactive mode
mtg-search

# Single search
mtg-search "red creature with haste"

# With streaming enabled (real-time AI thinking)
mtg-search --streaming "blue counterspell"

# Show example queries
mtg-search --examples
```

**Development mode (from source):**
```bash
# Interactive mode
python -m mtg_search_agent.cli

# Single search
python -m mtg_search_agent.cli "red creature with haste"

# With streaming enabled (real-time AI thinking)
python -m mtg_search_agent.cli --streaming "blue counterspell"

# Show example queries
python -m mtg_search_agent.cli --examples
```

### Python API

```python
from mtg_search_agent import SearchOrchestrator

# Create orchestrator
orchestrator = SearchOrchestrator(enable_streaming=False)

# Search for cards
result = await orchestrator.search("cheap red creature with haste")

# Display results
orchestrator.print_final_results(result)
```

### Event System (New API)

This project uses an event-driven API for progress updates. Listeners receive event-class instances and register by event_type string.

Key points:
- Emitters deliver BaseEvent subclasses (no dict payloads).
- Register listeners with: `orchestrator.events.on("event_type", callback)`.
- Callback signature: `def handler(event: SomeEvent) -> None`.

Minimal example:

```python
from mtg_search_agent import SearchOrchestrator
from mtg_search_agent.events import SearchStartedEvent, SearchCompletedEvent

orch = SearchOrchestrator()

def on_started(event: SearchStartedEvent):
   print(f"Search started (max {event.max_iterations}) for: {event.query}")

def on_completed(event: SearchCompletedEvent):
   print(f"Done in {event.total_duration}, avg score {event.final_score:.1f}")

orch.events.on("search_started", on_started)
orch.events.on("search_completed", on_completed)

# later inside an async context
# result = await orch.search("blue instant that counters spells and draws cards")
```

Common event types you can handle:
- search_started, search_completed
- iteration_started, iteration_completed
- query_generation_started, query_streaming_progress, query_generated
- scryfall_pagination_started, scryfall_page_fetched, scryfall_pagination_completed
- cards_found, cache_analyzed
- evaluation_strategy_selected, evaluation_started, evaluation_streaming_progress, evaluation_parallel_metrics, evaluation_completed
- final_results_display, error_occurred

## Project Structure

```
SearchAgent/
├── src/mtg_search_agent/   # Main package
│   ├── __init__.py        # Package entry point
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration and API settings
│   ├── orchestrator.py    # Main search coordination logic
│   ├── events.py          # Event handling system
│   ├── tags.json          # Complete MTG tag database (included with package)
│   ├── models/            # Data models
│   │   ├── card.py       # Card data structures
│   │   ├── search.py     # Search query/result models
│   │   └── evaluation.py # Evaluation result models
│   ├── tools/            # Search tools
│   │   ├── tag_search.py # Fuzzy tag matching
│   │   └── scryfall_api.py # Scryfall API integration
│   └── agents/           # AI agents
│       ├── query_agent.py # Natural language → Scryfall query
│       └── evaluation_agent.py # Card relevance scoring
├── tests/               # Test suite
├── pyproject.toml       # Package configuration
└── README.md           # This file
```

## How It Works

1. **Query Agent** converts your natural language request into Scryfall search syntax
2. **Tag Search Tool** helps find valid Scryfall tags using fuzzy matching
3. **Scryfall API** searches for matching cards
4. **Evaluation Agent** scores each card's relevance (1-10) to your original request
5. **Orchestrator** manages up to 5 refinement loops until good results are found

Behind the scenes, the orchestrator emits events at each step so UIs (like the CLI) can display progress without coupling to core logic.

## Example Queries

- "I want a cheap red creature with haste that can deal damage quickly"
- "Find me a blue instant that can counter spells and draw cards"
- "I need a green ramp spell that puts lands into play"
- "Show me artifacts that cost 2 mana and provide mana acceleration"

## Building the Package

To build the package for distribution:

```bash
# Install build tools
pip install build

# Build the package
python -m build

# This creates dist/ directory with wheel and source distribution
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd SearchAgent

# Install in development mode with dependencies
pip install -e .

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running Tests

```bash
# Run the CLI to test functionality
python -m mtg_search_agent.cli --examples
python -m mtg_search_agent.cli "test query"

# Run unit tests
pytest
```

### Project Architecture

The project follows a clean, modular architecture:

- **Orchestrator**: Coordinates the multi-agent search process
- **Query Agent**: Converts natural language to Scryfall queries using GPT
- **Evaluation Agent**: Scores card relevance using GPT
- **Tools**: Scryfall API integration and tag fuzzy matching
- **Models**: Pydantic data structures for type safety
- **Events**: Event-driven progress tracking and display (event-class-only API)

### Breaking Changes (for upgraders)

- The legacy event enum and factory functions were removed. Use event classes and string event types instead:
   - Before: `events.on(SearchEventType.SEARCH_STARTED, handler)` and `emit(SearchEventType.X, create_*_event(...))`
   - Now: `events.on("search_started", handler)` and `emit(SearchStartedEvent(...))`

## API Rate Limiting

The system respects Scryfall's API guidelines with 100ms delays between requests.

## Dependencies

- **pydantic-ai**: AI agent framework
- **openai**: GPT model access
- **requests**: HTTP client for Scryfall API
- **python-dotenv**: Environment variable management
- **fuzzywuzzy**: Fuzzy string matching for tags
- **rich**: Beautiful CLI output formatting
- **click**: Command-line interface framework (optional)