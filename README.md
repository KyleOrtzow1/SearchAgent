# MTG Card Search Agent

A multi-agent system for finding Magic: The Gathering cards using natural language queries. Built with PydanticAI and powered by OpenAI.

## Features

- **Natural Language Processing**: Convert plain English requests into Scryfall search syntax
- **Multi-Agent Architecture**: Query generation and evaluation agents work together
- **Tag Similarity Search**: Fuzzy matching against comprehensive MTG tag database
- **Intelligent Iteration**: Up to 5 search refinement loops with agent feedback
- **Relevance Scoring**: 1-10 scoring system for card relevance

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API Key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the Example**:
   ```bash
   python example.py
   ```

## Project Structure

```
SearchAgent/
├── config.py              # Configuration and API settings
├── orchestrator.py        # Main search coordination logic
├── example.py            # Interactive demo script
├── models/              # Data models
│   ├── card.py         # Card data structures
│   ├── search.py       # Search query/result models
│   └── evaluation.py   # Evaluation result models
├── tools/              # Search tools
│   ├── tag_search.py   # Fuzzy tag matching
│   └── scryfall_api.py # Scryfall API integration
├── agents/             # AI agents
│   ├── query_agent.py  # Natural language → Scryfall query
│   └── evaluation_agent.py # Card relevance scoring
└── tags.json          # Complete MTG tag database
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

## API Rate Limiting

The system respects Scryfall's API guidelines with 100ms delays between requests.