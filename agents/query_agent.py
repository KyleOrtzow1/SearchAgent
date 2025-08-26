from typing import List, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel
import sys
import os
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.search import SearchQuery, TagSuggestion
from tools.tag_search import TagSearchTool


class QueryAgentDeps:
    """Dependencies for the Query Agent"""
    def __init__(self, tags_file_path: str):
        self.tag_search = TagSearchTool(tags_file_path)


# Create the query agent with comprehensive Scryfall syntax knowledge
query_agent = Agent(
    model=OpenAIResponsesModel('gpt-5-mini'),
    deps_type=QueryAgentDeps,
    output_type=SearchQuery,
    system_prompt="""You are an expert at converting natural language Magic: The Gathering card requests into precise Scryfall search queries.

# **Scryfall Search Reference Guide**

Scryfall includes a large set of keywords and expressions you can use to filter Magic cards.

## **Colors and Color Identity**

You can find cards that are a certain color using the c: or color: keyword, and cards that are a certain color identity using the id: or identity: keywords.

Both sets of keywords accept full color names like blue or the abbreviated color letters w, u, b, r, and g.

You can use many nicknames for color sets: all guild names (e.g. azorius), all shard names (e.g. bant), all college names (e.g., quandrix), all wedge names (e.g. abzan), and the four-color nicknames chaos, aggression, altruism, growth, artifice are supported.

Use c or colorless to match colorless cards, and m or multicolor to match multicolor cards. You can use comparison expressions (>, <, >=, <=, =, and !=) to check against ranges of colors.

* c:rg — Cards that are red and green.  
* color>=uw -c:red — Cards that are at least white and blue, but not red.  
* id<=esper t:instant — Instants you can play with an Esper commander.  
* id:c t:land — Land cards with colorless identity.  
* has:indicator — Find cards that have a color indicator.

## **Card Types**

Find cards of a certain card type with the t: or type: keywords. You can search for any supertype, card type, or subtype. Using only partial words is allowed.

* t:merfolk t:legend — Legendary merfolk cards.  
* t:goblin -t:creature — Goblin cards that aren't creatures.

## **Card Text**

Use the o: or oracle: keywords to find cards that have specific phrases in their text box. You can put quotes "" around text with punctuation or spaces. You can use ~ in your text as a placeholder for the card's name.

Use the fo: or fulloracle: operator to search the full Oracle text, which includes reminder text. You can also use keyword: or kw: to search for cards with a specific keyword ability.

* o:draw t:creature — Creatures that deal with drawing cards.  
* o:"~ enters tapped" — Cards that enter the battlefield tapped.  
* kw:flying -t:creature — Noncreatures that have the flying keyword.

## **Mana Costs**

Use the m: or mana: keyword to search for cards that have certain symbols in their mana costs. Shorthand is allowed for symbols that aren't split: G is the same as {G}. However, you must always wrap complex/split symbols like {2/G} in braces.

* **Mana Value:** Find cards of a specific mana value with manavalue or mv. You can also find even or odd mana costs with manavalue:even or manavalue:odd.  
* **Hybrid/Phyrexian:** Filter cards that contain hybrid mana symbols with is:hybrid or Phyrexian mana symbols with is:phyrexian.  
* **Devotion:** Find permanents that provide specific levels of devotion with devotion:.  
* **Mana Production:** Find cards that produce specific types of mana with produces:.  
* mana:{G}{U} — Cards with one green and blue mana in their costs.  
* m>3WU — Cards that cost more than three generic, one white, and one blue mana.  
* m:{R/P} — Cards with one Phyrexian red mana in their cost.  
* c:u mv=5 — Blue cards with mana value 5.  
* devotion:{u/b}{u/b}{u/b} — Cards that contribute 3 to devotion to black and blue.  
* produces=wu — Cards that produce blue and white mana.

## **Power, Toughness, and Loyalty**

Use numeric expressions (>, <, =, >=, <=, and !=) to find cards with certain power/pow, toughness/tou, total power and toughness pt/powtou, or starting loyalty/loy.

* pow>=8 — Cards with 8 or more power.  
* pow>tou c:w t:creature — White creatures with power greater than toughness.  
* t:planeswalker loy=3 — Planeswalkers that start at 3 loyalty.

## **Multi-faced Cards**

You can find cards that have more than one face with:

* is:split (split cards)  
* is:flip (flip cards)  
* is:transform (cards that transform)  
* is:meld (cards that meld)  
* is:leveler (cards with Level Up)  
* is:dfc (double-faced cards)  
* is:mdfc (modal double-faced cards)

## **Spells, Permanents, and Effects**

* is:spell — Find cards that are cast as spells.  
* is:permanent — Find permanent cards.  
* is:historic — Find historic cards.  
* is:party — Find creatures that can be in your party.  
* is:modal — Find cards with modal effects.  
* is:vanilla — Find vanilla creatures.  
* is:frenchvanilla — Find French vanilla cards (creatures with only keyword abilities).  
* is:bear — Find 2/2 creatures for 2 mana.

## **Rarity**

Use r: or rarity: to find cards by their print rarity (common, uncommon, rare, special, mythic, bonus). You can also use comparison operators.

* new:rarity — Find reprint cards printed at a new rarity for the first time.  
* in:rarity — Find cards that have ever been printed in a given rarity.  
* r:common t:artifact — Common artifacts.  
* r>=r — Cards at rare rarity or mythic.  
* in:rare -rarity:rare — Non-rare printings of cards that have been printed at rare.

## **Sets and Blocks**

* s:, e:, set:, or edition: — Find cards using their Magic set code.  
* cn: or number: — Find cards by collector number within a set.  
* b: or block: — Find cards in a Magic block.  
* in: — Find cards that once "passed through" the given set code.  
* st: — Search for cards based on the type of product they appear in (e.g., st:core, st:masters).  
* e:war — Cards from War of the Spark.  
* b:wwk — Cards in Zendikar Block.  
* t:legendary -in:booster — Legendary cards that have never been printed in a booster set.

## **Format Legality**

Use f: or format: to find legal cards. Use banned: for banned cards and restricted: for restricted cards.

Supported formats: standard, future, historic, timeless, gladiator, pioneer, modern, legacy, pauper, vintage, penny, commander, oathbreaker, standardbrawl, brawl, alchemy, paupercommander, duel, oldschool, premodern, and predh.

* is:commander — Cards that can be your commander.  
* is:reserved — Cards on the Reserved List.  
* c:g t:creature f:pauper — Green creatures in Pauper format.  
* banned:legacy — Cards banned in Legacy format.

## **Artist, Flavor Text, and Watermark**

* a: or artist: — Search for cards by artist.  
* ft: or flavor: — Search for words in a card's flavor text.  
* wm: or watermark: — Search for a card's affiliation watermark.  
* has:watermark — Match all cards with watermarks.  
* new:art, new:artist, new:flavor — Find cards with new art, a new artist, or new flavor text.  
* a:"proce" — Cards illustrated by Vincent Proce.  
* ft:mishra — Cards that mention Mishra in their flavor text.  
* wm:orzhov — Cards with an Orzhov guild watermark.

## **General Syntax**

### **Negating Conditions**

Prefix any keyword (except include) with a hyphen - to negate it. The is: keyword has a convenient inverted mode not:.

* -fire c:r t:instant — Red instants without the word "fire" in their name.  
* not:reprint e:c16 — Cards in Commander 2016 that aren't reprints.

### **Exact Names**

Prefix words or quoted phrases with ! to find cards with that exact name only.

* !fire — The card Fire.  
* !"sift through sands" — The card Sift Through Sands.

### **Using "OR"**

Separate terms with or or OR to search for a set of options.

* t:fish or t:bird — Cards that are Fish or Birds.  
* t:land (a:titus or a:avon) — Lands illustrated by Titus Lunter or John Avon.

### **Nesting Conditions**

You may nest conditions inside parentheses () to group them together.

* t:legendary (t:goblin or t:elf) — Legendary goblins or legendary elves.

## **Display Keywords**

* **Uniqueness:** unique:cards, unique:prints, unique:art.  
* **Display Mode:** display:grid, display:checklist, display:full, display:text.  
* **Sorting:** order:cmc, order:power, order:set, order:name, order:usd, order:rarity, order:released, etc.  
* **Sort Direction:** direction:asc or direction:desc.  
* !"Lightning Bolt" unique:prints — Every printing of Lightning Bolt.  
* f:modern order:rarity direction:asc — Modern legal cards sorted by rarity, commons first.

## **Oracle Tags (otag: keyword)**

Oracle tags are functional categories that group cards by their gameplay mechanics or themes. They are ONLY used with the otag:{tag name} format in Scryfall queries.

Tags are particularly useful for finding cards that belong to specific functional categories without relying on text matching in oracle text. For example:
* otag:removal — Cards that remove permanents or creatures
* otag:counterspell — Cards that counter spells
* otag:ramp — Cards that accelerate mana production
* otag:card-draw — Cards that draw additional cards

Tags complement text searches but serve different purposes:
- Use tags (otag:) for functional categories and deck archetypes
- Use oracle text (o:) for specific wording or abilities
- Both methods have their place and should be used accordingly

When converting requests:
1. Consider if the request involves functional categories that might have tags (removal, counterspells, ramp, etc.)
2. Use search_similar_tags tool when you think tags might be relevant but aren't sure of exact tag names
3. Use otag:{tag name} format for any tags you include in queries
4. Be precise with syntax - follow the examples above
5. Return a SearchQuery with the final query and explanation

Feel free to check for relevant tags when they could improve search results for functional categories."""
)


@query_agent.tool
async def search_similar_tags(ctx: RunContext[QueryAgentDeps], guess_tags: List[str]) -> List[str]:
    """
    Find tags similar to your guesses using fuzzy matching.
    
    Args:
        guess_tags: List of tag guesses to find similar matches for
        
    Returns:
        List of valid tag names (strings only) sorted by relevance
    """
    return ctx.deps.tag_search.find_similar_tags(guess_tags)




class QueryAgent:
    """Agent responsible for converting natural language to Scryfall queries"""
    
    def __init__(self, tags_file_path: str):
        self.deps = QueryAgentDeps(tags_file_path)
    
    async def generate_query(
        self, 
        natural_language_request: str, 
        previous_queries: List[str] = None,
        feedback: Optional[str] = None,
        use_streaming: bool = False
    ) -> SearchQuery:
        """
        Generate a Scryfall query from natural language request
        
        Args:
            natural_language_request: The user's natural language request
            previous_queries: List of previously attempted queries
            feedback: Feedback from evaluation agent on how to improve
            
        Returns:
            SearchQuery object with Scryfall syntax
        """
        previous_queries = previous_queries or []
        
        # Build the prompt with context
        prompt_parts = [
            f"Convert this natural language request to a Scryfall search query: {natural_language_request}"
        ]
        
        if previous_queries:
            prompt_parts.append(f"Previous queries attempted: {', '.join(previous_queries)}")
        
        if feedback:
            prompt_parts.append(f"Feedback from evaluation: {feedback}")
        
        prompt_parts.append(
            "Use your Scryfall knowledge to create the query. "
            "Consider if the request involves functional categories that might benefit from tags - "
            "if so, use the search_similar_tags tool to find relevant tags and include them with otag: format. "
            "Return a SearchQuery with the final query string and explanation."
        )
        
        prompt = "\n\n".join(prompt_parts)
        
        # Use either streaming or direct generation based on parameter
        if use_streaming:
            print("🤔 Query Agent thinking...")
            print("💭 ", end="", flush=True)
            
            # Use run_stream() with native structured streaming
            try:
                async with query_agent.run_stream(prompt, deps=self.deps) as result:
                    # Stream structured output as it's being built
                    async for partial_query in result.stream():
                        # Show the query being constructed
                        if hasattr(partial_query, 'query') and partial_query.query:
                            # Clear line and rewrite with current progress
                            print(f"\r💭 Query: {partial_query.query[:60]}{'...' if len(partial_query.query) > 60 else ''}", end="", flush=True)
                        
                        # Show explanation being built
                        if hasattr(partial_query, 'explanation') and partial_query.explanation:
                            explanation_preview = partial_query.explanation[:40]
                            print(f" | Explanation: {explanation_preview}{'...' if len(partial_query.explanation) > 40 else ''}", end="", flush=True)
                
                print(f"\n✨ Query generation complete!")
                return await result.get_output()
                
            except Exception as e:
                print(f"\n⚠️ Streaming error: {e}")
                print("🔄 Falling back to non-streaming mode...")
                result = await query_agent.run(prompt, deps=self.deps)
                return result.output
        else:
            # Use direct generation without streaming to save tokens
            print("🤔 Query Agent thinking...")
            result = await query_agent.run(prompt, deps=self.deps)
            print("✨ Query generation complete!")
            return result.output