#!/usr/bin/env python3
"""
MTG Card Search CLI
A clean, user-friendly command-line interface for searching Magic: The Gathering cards.
"""

import asyncio
import os
import sys
from typing import Optional, List
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.status import Status
from rich.columns import Columns
from rich.rule import Rule
from rich.tree import Tree
from rich import box
from datetime import datetime
import time
from dotenv import load_dotenv

from orchestrator import SearchOrchestrator
from models.evaluation import EvaluationResult
from events import SearchEventType

# Load environment variables
load_dotenv()

console = Console()


class MTGSearchCLI:
    """Main CLI application class with animated status display"""
    
    def __init__(self, tags_file: str = "tags.json", streaming: bool = False):
        self.tags_file = tags_file
        self.streaming = streaming
        self.orchestrator = None
        self.live_display = None
        self.layout = None
        
        # Status tracking
        self.search_history = []
        self.current_iteration = 1
        self.max_iterations = 5
        self.current_query = ""
        self.cards_found = 0
        self.evaluation_progress = 0
        self.evaluation_total = 0
        self.current_score = 0.0
        self.best_score = 0.0
        self.start_time = None
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator and check dependencies"""
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
            console.print("Please create a .env file with your OpenAI API key:")
            console.print("[cyan]OPENAI_API_KEY=your_api_key_here[/cyan]")
            return False
            
        if not Path(self.tags_file).exists():
            console.print(f"[red]Error: Tags file '{self.tags_file}' not found.[/red]")
            return False
            
        try:
            self.orchestrator = SearchOrchestrator(self.tags_file, enable_streaming=self.streaming)
            self._setup_event_handlers()
            return True
        except Exception as e:
            console.print(f"[red]Error initializing orchestrator: {e}[/red]")
            return False

    def _setup_event_handlers(self):
        """Set up event handlers for clean progress display"""
        
        def on_search_started(data):
            self.search_status = Status("[cyan]Starting search...[/cyan]")
            self.search_status.start()
            
        def on_iteration_started(data):
            if self.search_status:
                self.search_status.update(f"[cyan]Iteration {data['iteration']}/{data['max_iterations']}[/cyan]")
        
        def on_query_generated(data):
            if self.search_status:
                query = data['scryfall_query']
                # Truncate long queries for display
                display_query = query[:50] + "..." if len(query) > 50 else query
                self.search_status.update(f"[blue]Generated query: {display_query}[/blue]")
        
        def on_cards_found(data):
            if self.search_status:
                count = data['count']
                self.search_status.update(f"[green]Found {count} cards[/green]")
        
        def on_evaluation_started(data):
            if self.search_status:
                count = data['card_count']
                self.search_status.update(f"[yellow]Evaluating {count} cards...[/yellow]")
        
        def on_evaluation_progress(data):
            if self.search_status:
                percent = data.get('progress_percent', 0)
                self.search_status.update(f"[yellow]Evaluating cards... {percent:.0f}% complete[/yellow]")
        
        def on_evaluation_completed(data):
            if self.search_status:
                score = data['average_score']
                self.search_status.update(f"[magenta]Evaluation complete (avg score: {score:.1f}/10)[/magenta]")
        
        def on_iteration_completed(data):
            if self.search_status:
                score = data['score']
                iteration = data['iteration']
                if data['is_best']:
                    self.search_status.update(f"[bright_green]Iteration {iteration} complete! New best score: {score:.1f}/10[/bright_green]")
                else:
                    self.search_status.update(f"[green]Iteration {iteration} complete (score: {score:.1f}/10)[/green]")
        
        def on_search_completed(data):
            if self.search_status:
                self.search_status.stop()
                self.search_status = None
        
        def on_query_generation_started(data):
            if self.search_status:
                self.search_status.update("[blue]Generating search query...[/blue]")
        
        def on_query_streaming_progress(data):
            if self.search_status:
                query = data.get('partial_query', '')
                explanation = data.get('partial_explanation', '')
                if query:
                    display_query = query[:50] + "..." if len(query) > 50 else query
                    self.search_status.update(f"[blue]Building query: {display_query}[/blue]")
        
        def on_scryfall_pagination_started(data):
            if self.search_status:
                query = data['query']
                display_query = query[:50] + "..." if len(query) > 50 else query
                self.search_status.update(f"[cyan]Fetching complete results for: {display_query}[/cyan]")
        
        def on_scryfall_page_fetched(data):
            if self.search_status:
                page = data['page']
                cards_so_far = data['total_cards_so_far']
                total_available = data['total_available']
                progress = data.get('progress_percent', 0)
                self.search_status.update(f"[cyan]Fetched page {page}: {cards_so_far}/{total_available} cards ({progress:.0f}%)[/cyan]")
        
        def on_scryfall_pagination_completed(data):
            if self.search_status:
                total_cards = data['total_cards']
                pages = data['pages_fetched']
                limited = data['limited_by_max']
                status_text = f"[green]Fetched {total_cards} cards from {pages} page(s)[/green]"
                if limited:
                    status_text += " [yellow](limited by max results)[/yellow]"
                self.search_status.update(status_text)
        
        def on_evaluation_streaming_progress(data):
            if self.search_status:
                evaluated = data['cards_evaluated']
                total = data['total_cards']
                progress = data.get('progress_percent', 0)
                batch_info = data.get('batch_index'), data.get('total_batches')
                
                if batch_info[0] is not None and batch_info[1] is not None:
                    batch_text = f"Batch {batch_info[0] + 1}/{batch_info[1]}: "
                else:
                    batch_text = ""
                
                score_text = ""
                if data.get('current_score'):
                    score_text = f" (score: {data['current_score']:.1f})"
                
                self.search_status.update(f"[yellow]{batch_text}Evaluating {evaluated}/{total} cards ({progress:.0f}%){score_text}[/yellow]")
        
        def on_error_occurred(data):
            if self.search_status:
                error_type = data.get('error_type', 'unknown')
                message = data.get('message', 'An error occurred')
                # Show brief error message
                self.search_status.update(f"[red]⚠️ {error_type}: {message[:50]}{'...' if len(message) > 50 else ''}[/red]")
        
        def on_evaluation_strategy_selected(data):
            if self.search_status:
                strategy = data.get('strategy', 'unknown')
                card_count = data.get('card_count', 0)
                reason = data.get('reason', '')
                
                if strategy == "parallel_batch":
                    batch_size = data.get('batch_size', 0)
                    self.search_status.update(f"[cyan]🚀 Parallel evaluation: {card_count} cards (batch size: {batch_size})[/cyan]")
                elif strategy == "parallel_execution":
                    total_batches = data.get('total_batches', 0)
                    batch_size = data.get('batch_size', 0)
                    self.search_status.update(f"[cyan]⚡ Processing {card_count} cards in {total_batches} batches of {batch_size}[/cyan]")
                elif strategy == "bulk_evaluation":
                    self.search_status.update(f"[cyan]📊 Bulk evaluation: {card_count} cards ({reason})[/cyan]")
        
        def on_evaluation_parallel_metrics(data):
            if self.search_status:
                total_batches = data.get('total_batches', 0)
                elapsed_time = data.get('elapsed_time', 0)
                time_saved = data.get('time_saved')
                
                status_text = f"[green]⚡ All {total_batches} batches completed in {elapsed_time:.1f}s[/green]"
                if time_saved and time_saved > 0:
                    status_text += f" [bright_green](saved {time_saved:.1f}s)[/bright_green]"
                self.search_status.update(status_text)
        
        def on_final_results_display(data):
            """Handle final results display event"""
            if not data.get('has_results', False):
                console.print("[yellow]No relevant cards found.[/yellow]")
                return
            
            # Display results header
            console.print("\n" + "="*60)
            console.print("[bold]FINAL SEARCH RESULTS[/bold]")
            console.print("="*60)
            
            # Display summary information
            iteration_count = data.get('iteration_count', 1)
            total_unique_cards = data.get('total_unique_cards_evaluated', 0)
            average_score = data.get('average_score', 0.0)
            total_cards = data.get('total_cards', 0)
            
            console.print(f"Search completed in {iteration_count} iteration(s)")
            if total_unique_cards > 0:
                console.print(f"Total unique cards evaluated: {total_unique_cards}")
            console.print(f"Showing top {total_cards} highest scoring cards (Average: {average_score:.1f}/10)")
            console.print()
            
            # Display individual cards
            for index, card_data in enumerate(data.get('scored_cards', []), 1):
                console.print(f"{index}. {card_data['name']} - Score: {card_data['score']}/10")
                console.print(f"   Mana Cost: {card_data['mana_cost'] or 'None'}")
                console.print(f"   Type: {card_data['type_line']}")
                
                if card_data.get('power') and card_data.get('toughness'):
                    console.print(f"   Power/Toughness: {card_data['power']}/{card_data['toughness']}")
                
                if card_data.get('reasoning'):
                    console.print(f"   Reasoning: {card_data['reasoning']}")
                    
                console.print(f"   Scryfall: {card_data['scryfall_uri']}")
                console.print()
        
        # Register event handlers
        self.orchestrator.events.on(SearchEventType.SEARCH_STARTED, on_search_started)
        self.orchestrator.events.on(SearchEventType.ITERATION_STARTED, on_iteration_started)
        self.orchestrator.events.on(SearchEventType.QUERY_GENERATION_STARTED, on_query_generation_started)
        self.orchestrator.events.on(SearchEventType.QUERY_STREAMING_PROGRESS, on_query_streaming_progress)
        self.orchestrator.events.on(SearchEventType.QUERY_GENERATED, on_query_generated)
        self.orchestrator.events.on(SearchEventType.SCRYFALL_PAGINATION_STARTED, on_scryfall_pagination_started)
        self.orchestrator.events.on(SearchEventType.SCRYFALL_PAGE_FETCHED, on_scryfall_page_fetched)
        self.orchestrator.events.on(SearchEventType.SCRYFALL_PAGINATION_COMPLETED, on_scryfall_pagination_completed)
        self.orchestrator.events.on(SearchEventType.CARDS_FOUND, on_cards_found)
        self.orchestrator.events.on(SearchEventType.EVALUATION_STRATEGY_SELECTED, on_evaluation_strategy_selected)
        self.orchestrator.events.on(SearchEventType.EVALUATION_STARTED, on_evaluation_started)
        self.orchestrator.events.on(SearchEventType.EVALUATION_STREAMING_PROGRESS, on_evaluation_streaming_progress)
        self.orchestrator.events.on(SearchEventType.EVALUATION_PARALLEL_METRICS, on_evaluation_parallel_metrics)
        self.orchestrator.events.on(SearchEventType.EVALUATION_BATCH_PROGRESS, on_evaluation_progress)
        self.orchestrator.events.on(SearchEventType.EVALUATION_COMPLETED, on_evaluation_completed)
        self.orchestrator.events.on(SearchEventType.ITERATION_COMPLETED, on_iteration_completed)
        self.orchestrator.events.on(SearchEventType.SEARCH_COMPLETED, on_search_completed)
        self.orchestrator.events.on(SearchEventType.FINAL_RESULTS_DISPLAY, on_final_results_display)
        self.orchestrator.events.on(SearchEventType.ERROR_OCCURRED, on_error_occurred)

    def show_banner(self):
        """Display the application banner"""
        banner = Panel(
            Align.center(
                Text("MTG Card Search Agent", style="bold blue") + "\n" +
                Text("Powered by AI agents and Scryfall API", style="dim")
            ),
            border_style="blue",
            padding=(1, 2)
        )
        console.print(banner)

    def format_card_results(self, result: EvaluationResult) -> Table:
        """Format search results into a rich table"""
        table = Table(
            title="Search Results",
            border_style="green",
            header_style="bold green"
        )
        
        table.add_column("Rank", justify="center", style="cyan", width=4)
        table.add_column("Score", justify="center", style="yellow", width=5)
        table.add_column("Card Name", style="bold white", min_width=20)
        table.add_column("Cost", justify="center", style="magenta", width=8)
        table.add_column("Type", style="blue", width=15)
        table.add_column("Reason", style="dim white", max_width=40)
        
        for i, scored_card in enumerate(result.scored_cards, 1):
            card = scored_card.card
            score = scored_card.score
            
            # Format mana cost
            mana_cost = card.mana_cost or "N/A"
            
            # Format card type
            card_type = card.type_line or "Unknown"
            if len(card_type) > 15:
                card_type = card_type[:12] + "..."
            
            # Format reasoning
            reason = scored_card.reasoning or ""
            if len(reason) > 40:
                reason = reason[:37] + "..."
            
            table.add_row(
                str(i),
                f"{score}/10",
                card.name,
                mana_cost,
                card_type,
                reason
            )
        
        return table

    def show_search_summary(self, result: EvaluationResult):
        """Display search summary information"""
        summary_text = []
        summary_text.append(f"Found {len(result.scored_cards)} relevant cards")
        if hasattr(result, 'iteration_count') and result.iteration_count > 1:
            summary_text.append(f"Refined search {result.iteration_count} times")
        if hasattr(result, 'average_score') and result.average_score:
            summary_text.append(f"Average relevance: {result.average_score:.1f}/10")
            
        summary = Panel(
            "\n".join(summary_text),
            title="Search Summary",
            border_style="dim",
            padding=(0, 1)
        )
        console.print(summary)

    async def interactive_search(self):
        """Run interactive search mode"""
        console.print("\n[bold]Interactive Search Mode[/bold]")
        console.print("Type your card search requests naturally, or 'quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("[cyan]Search for cards")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query.strip():
                    console.print("[yellow]Please enter a search query.[/yellow]")
                    continue
                
                # Perform search (progress is handled by events now)
                result = await self.orchestrator.search(query)
                
                # Display results
                if result.scored_cards:
                    console.print()
                    self.show_search_summary(result)
                    console.print()
                    console.print(self.format_card_results(result))
                else:
                    console.print("[yellow]No matching cards found. Try a different search.[/yellow]")
                
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Search cancelled.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error during search: {e}[/red]")

    async def single_search(self, query: str):
        """Perform a single search and display results"""
        try:
            # Perform search (progress is handled by events now)
            result = await self.orchestrator.search(query)
            
            if result.scored_cards:
                console.print()
                self.show_search_summary(result)
                console.print()
                console.print(self.format_card_results(result))
            else:
                console.print("[yellow]No matching cards found.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error during search: {e}[/red]")
            sys.exit(1)

    def show_examples(self):
        """Display example search queries"""
        examples = [
            "cheap red creature with haste",
            "blue instant that counters spells and draws cards", 
            "green ramp spell that puts lands into play",
            "artifacts that cost 2 mana and provide mana acceleration",
            "legendary creatures good for token commanders"
        ]
        
        console.print("\n[bold]Example Search Queries:[/bold]")
        for i, example in enumerate(examples, 1):
            console.print(f"  {i}. [dim]{example}[/dim]")
        console.print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MTG Card Search Agent - AI-powered Magic card search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s "red creature with haste"          # Single search
  %(prog)s --streaming "blue counterspell"    # Enable AI streaming
  %(prog)s --examples                         # Show example queries
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Card search query (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable real-time AI streaming (uses more tokens)"
    )
    
    parser.add_argument(
        "--tags-file",
        default="tags.json",
        help="Path to tags file (default: tags.json)"
    )
    
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show example search queries and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MTG Search Agent 1.0"
    )
    
    args = parser.parse_args()
    
    # Handle special cases
    if args.examples:
        cli = MTGSearchCLI()
        cli.show_examples()
        return
    
    async def run_cli():
        cli = MTGSearchCLI(args.tags_file, args.streaming)
        
        # Show banner
        cli.show_banner()
        
        # Initialize
        if not await cli.initialize():
            sys.exit(1)
        
        # Show streaming mode info
        if args.streaming:
            console.print("[dim]Streaming mode enabled - real-time AI thinking[/dim]\n")
        
        # Run appropriate mode
        if args.query:
            # Single search mode
            console.print(f"[bold]Searching for:[/bold] {args.query}\n")
            await cli.single_search(args.query)
        else:
            # Interactive mode
            await cli.interactive_search()
        
        console.print("\n[dim]Thanks for using MTG Search Agent![/dim]")
    
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()