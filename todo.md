# MTG Search Agent - TODO List

## Bug Fixes

## Configuration Improvements

## Feedback System Improvements

## Search Result Improvements

## Query Agent Improvements
- [ ] Improve color identity understanding - query agent should understand that color identity (id:) means cards that fit "under" a commander of that color (e.g., id:white includes both white cards and colorless cards)

## Completed
- [X] Implement pagination to return all cards matching a query instead of being limited to ~175 cards (Scryfall API limitation) - add ability to fetch complete result sets
- [X] Extract stop loop confidence threshold to config.py so it can be changed on the fly (currently hardcoded as `< 6` in evaluation logic)
- [X] Replace hardcoded text extraction feedback aggregation with LLM-based feedback synthesis - take all batch feedback and use LLM to generate final combined feedback instead of keyword matching
- [X] Fix query agent tag usage - ensure found tags are actually used in `otag:{tag}` format in searches (currently finding tags but not using them)
- [X] Fix eval agent batching issue - provide total card count across all batches so eval agents understand the full search context instead of just their batch
- [X] Implement feedback aggregation from batched eval agents to query agent - combine all batch feedback into summary for query refinement
- [X] Implement card caching system to avoid re-evaluating same cards
- [X] Add parallel evaluation for multiple cards
- [X] Add streaming support to see agent reasoning in real-time
- [X] Test performance improvements with caching and streaming