# MTG Search Agent - TODO List

## Bug Fixes
- [X] Fix eval agent batching issue - provide total card count across all batches so eval agents understand the full search context instead of just their batch
- [ ] Fix query agent tag usage - ensure found tags are actually used in `otag:{tag}` format in searches (currently finding tags but not using them)
- [X] Implement feedback aggregation from batched eval agents to query agent - combine all batch feedback into summary for query refinement

## Completed
- [X] Implement card caching system to avoid re-evaluating same cards
- [X] Add parallel evaluation for multiple cards
- [X] Add streaming support to see agent reasoning in real-time
- [X] Test performance improvements with caching and streaming