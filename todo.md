# MTG Search Agent - TODO List

## Bug Fixes
- [X] Change caching to track by card ID instead of name - modify card caching system to use Scryfall card IDs instead of card names for better accuracy and handling of cards with same names

## Configuration Improvements

## Feedback System Improvements

## Search Result Improvements

## Query Agent Improvements
- [ ] Improve color identity understanding - query agent should understand that color identity (id:) means cards that fit "under" a commander of that color (e.g., id:white includes both white cards and colorless cards)

## Code Refactoring
- [X] Refactor models to check for redundancy and streamline - review all Pydantic models for duplicate fields, unused classes, and opportunities to consolidate similar structures
- [ ] Refactor query agent for better streamlining - review query agent code structure, prompt organization, and method efficiency to make it more maintainable and performant
- [ ] Refactor orchestrator for better organization - review orchestrator code structure, method sizes, and separation of concerns to improve readability and maintainability

## Interface Development
- [ ] Build proper user interface to replace example script - design and implement a more user-friendly interface (web UI, GUI, or improved CLI) instead of relying on the basic example.py script
- [ ] Refactor project for dual-mode usage - enable the system to work both as a standalone user application AND as a tool that can be called by higher-level agents or other systems programmatically

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