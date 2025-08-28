# MTG Card Search Agent

A multi-agent system for finding Magic: The Gathering cards using natural language queries. Built with PydanticAI and powered by OpenAI.

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

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Set up OpenAI API Key**:
   ```bash
   # Create .env file with your API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
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

```bash
# Interactive mode
python examples/cli_demo.py

# Single search
python examples/cli_demo.py "red creature with haste"

# With streaming enabled (real-time AI thinking)
python examples/cli_demo.py --streaming "blue counterspell"

# Show example queries
python examples/cli_demo.py --examples
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

## Project Structure

```
SearchAgent/
├── src/mtg_search_agent/   # Main package
│   ├── __init__.py        # Package entry point
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
├── examples/             # Usage examples
│   └── cli_demo.py      # Interactive CLI demo
├── pyproject.toml       # Package configuration
└── README.md           # This file
```

## How It Works

1. **Query Agent** converts your natural language request into Scryfall search syntax
2. **Tag Search Tool** helps find valid Scryfall tags using fuzzy matching
3. **Scryfall API** searches for matching cards
4. **Evaluation Agent** scores each card's relevance (1-10) to your original request
5. **Orchestrator** manages up to 5 refinement loops until good results are found

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
# Run the CLI demo to test functionality
python examples/cli_demo.py --examples
python examples/cli_demo.py "test query"
```

### Project Architecture

The project follows a clean, modular architecture:

- **Orchestrator**: Coordinates the multi-agent search process
- **Query Agent**: Converts natural language to Scryfall queries using GPT
- **Evaluation Agent**: Scores card relevance using GPT
- **Tools**: Scryfall API integration and tag fuzzy matching
- **Models**: Pydantic data structures for type safety
- **Events**: Event-driven progress tracking and display

## API Rate Limiting

The system respects Scryfall's API guidelines with 100ms delays between requests.

## Dependencies

- **pydantic-ai**: AI agent framework
- **openai**: GPT model access
- **requests**: HTTP client for Scryfall API
- **python-dotenv**: Environment variable management
- **fuzzywuzzy**: Fuzzy string matching for tags
- **rich**: Beautiful CLI output formatting
- **click**: Command-line interface framework