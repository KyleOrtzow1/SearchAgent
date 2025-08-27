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
        
        # Status tracking for animated display
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
        self.search_active = False
        self.current_step = "Ready"
        self.search_status = None
        self.current_status_text = ""
        # Throttling/deduping for streaming updates
        self._last_status_time = 0.0
        self._status_throttle_secs = 0.5
        self._last_query_stream_text = ""
        self._last_eval_progress_printed = -1
    def _update_status(self, status_text: str):
        """Update the current status display"""
        # Print only when it changes to reduce noise
        if status_text != self.current_status_text:
            console.print(f"[dim]Status:[/dim] {status_text}")
            self.current_status_text = status_text
    
    def _clear_status(self):
        """Clear the current status display"""
        # Just print a newline to separate from the results
        if self.current_status_text:
            console.print()
            self.current_status_text = ""
        
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
            self.start_time = time.time()
            self.search_active = True
            self.search_history = []
            self.max_iterations = data.get('max_iterations', 5)
            
            # Simple status display without spinners
            self._update_status("[cyan]Starting search...[/cyan]")
            
            # Track the first activity
            self.search_history.append("Search initialized")
            
        def on_iteration_started(data):
            self.current_iteration = data['iteration']
            self.search_history.append(f"-> Iteration {data['iteration']}/{data['max_iterations']}")
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            elapsed_str = f"{elapsed:.1f}s"
            
            # Show concise current status without repeating history
            status_text = f"[cyan]Iteration {data['iteration']}/{data['max_iterations']} ({elapsed_str})[/cyan]"
            self._update_status(status_text)
        
        def on_query_generated(data):
            self.current_query = data.get('scryfall_query', data.get('query', 'Unknown'))
            self.search_history.append(f"Query: {self.current_query}")
            status_text = f"[blue]Generated: {self.current_query}[/blue]"
            self._update_status(status_text)
        
        def on_cards_found(data):
            self.cards_found = data['count']
            self.search_history.append(f"Found {data['count']} cards")
            status_text = f"[green]Found {data['count']} cards[/green]"
            self._update_status(status_text)
        
        def on_evaluation_started(data):
            self.evaluation_total = data['card_count']
            self.evaluation_progress = 0
            self.search_history.append(f"Evaluating {data['card_count']} cards")
            status_text = f"[yellow]Evaluating {data['card_count']} cards...[/yellow]"
            self._update_status(status_text)
        
        def on_evaluation_progress(data):
            self.evaluation_progress = data.get('progress_percent', 0)
            # Throttle: print only on 5% changes
            current_pct = int(self.evaluation_progress)
            if current_pct // 5 != self._last_eval_progress_printed // 5:
                progress_bar = "=" * (current_pct // 10) + "-" * (10 - (current_pct // 10))
                status_text = f"[yellow]Evaluating... [{progress_bar}] {self.evaluation_progress:.0f}%[/yellow]"
                self._update_status(status_text)
                self._last_eval_progress_printed = current_pct
        
        def on_evaluation_completed(data):
            self.current_score = data['average_score']
            self.search_history.append(f"Score: {self.current_score:.1f}/10")
            
            status_text = f"[magenta]DONE Score: {self.current_score:.1f}/10[/magenta]"
            self._update_status(status_text)
        
        def on_iteration_completed(data):
            score = data['score']
            iteration = data['iteration']
            
            if data['is_best']:
                self.best_score = score
                self.search_history.append(f"NEW BEST: {score:.1f}/10")
                
                status_text = f"[bright_green]*** NEW BEST! {score:.1f}/10 ***[/bright_green]"
                self._update_status(status_text)
            else:
                self.search_history.append(f"Iteration {iteration}: {score:.1f}/10")
                status_text = f"[green]DONE Iteration {iteration}: {score:.1f}/10[/green]"
                self._update_status(status_text)
        
        def on_search_completed(data):
            self.search_active = False
            final_score = data.get('final_score', self.best_score)
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            # Add completion info to history
            self.search_history.append(f"COMPLETED! Final: {final_score:.1f}/10")
            
            # Display final status
            status_text = f"[bright_cyan]SEARCH COMPLETE! Final Score: {final_score:.1f}/10 ({elapsed:.1f}s)[/bright_cyan]"
            self._update_status(status_text)
            
            # Clear status after a brief moment
            import threading
            def clear_after_delay():
                time.sleep(0.5)
                self._clear_status()
                console.print()  # Add a newline after clearing
            
            threading.Thread(target=clear_after_delay, daemon=True).start()
        
        def on_query_generation_started(data):
            self._update_status("[blue]Generating search query...[/blue]")
        
        def on_query_streaming_progress(data):
            query = data.get('partial_query', '')
            explanation = data.get('partial_explanation', '')
            if query:
                # Throttle frequent updates and avoid repetition; show full query
                now = time.time()
                if (now - self._last_status_time) >= self._status_throttle_secs or query != self._last_query_stream_text:
                    self._update_status(f"[blue]Building query: {query}[/blue]")
                    self._last_status_time = now
                    self._last_query_stream_text = query
        
        def on_scryfall_pagination_started(data):
            query = data['query']
            self._update_status(f"[cyan]Fetching complete results for: {query}[/cyan]")
        
        def on_scryfall_page_fetched(data):
            page = data['page']
            cards_so_far = data['total_cards_so_far']
            total_available = data['total_available']
            progress = data.get('progress_percent', 0)
            self._update_status(f"[cyan]Fetched page {page}: {cards_so_far}/{total_available} cards ({progress:.0f}%)[/cyan]")
        
        def on_scryfall_pagination_completed(data):
            total_cards = data['total_cards']
            pages = data['pages_fetched']
            limited = data['limited_by_max']
            status_text = f"[green]Fetched {total_cards} cards from {pages} page(s)[/green]"
            if limited:
                status_text += " [yellow](limited by max results)[/yellow]"
            self._update_status(status_text)
        
        def on_evaluation_streaming_progress(data):
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
            
            self._update_status(f"[yellow]{batch_text}Evaluating {evaluated}/{total} cards ({progress:.0f}%){score_text}[/yellow]")
        
        def on_error_occurred(data):
            error_type = data.get('error_type', 'unknown')
            message = data.get('message', 'An error occurred')
            # Show brief error message
            self._update_status(f"[red]ERROR {error_type}: {message}[/red]")
        
        def on_evaluation_strategy_selected(data):
            strategy = data.get('strategy', 'unknown')
            card_count = data.get('card_count', 0)
            reason = data.get('reason', '')
            
            if strategy == "parallel_batch":
                batch_size = data.get('batch_size', 0)
                self._update_status(f"[cyan]PARALLEL: {card_count} cards (batch size: {batch_size})[/cyan]")
            elif strategy == "parallel_execution":
                total_batches = data.get('total_batches', 0)
                batch_size = data.get('batch_size', 0)
                self._update_status(f"[cyan]PROCESSING: {card_count} cards in {total_batches} batches of {batch_size}[/cyan]")
            elif strategy == "bulk_evaluation":
                self._update_status(f"[cyan]BULK EVAL: {card_count} cards ({reason})[/cyan]")
        
        def on_evaluation_parallel_metrics(data):
            total_batches = data.get('total_batches', 0)
            elapsed_time = data.get('elapsed_time', 0)
            time_saved = data.get('time_saved')
            
            status_text = f"[green]ALL {total_batches} batches completed in {elapsed_time:.1f}s[/green]"
            if time_saved and time_saved > 0:
                status_text += f" [bright_green](saved {time_saved:.1f}s)[/bright_green]"
            self._update_status(status_text)
        
        def on_final_results_display(data):
            """No-op to avoid duplicate final output; CLI renders results once."""
            return
        
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
        """Format search results into a rich table with Scryfall links"""
        table = Table(
            title="Search Results",
            border_style="green",
            header_style="bold green"
        )

        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Score", justify="center", style="yellow")
        table.add_column("Card Name", style="bold white")
        table.add_column("Cost", justify="center", style="magenta")
        table.add_column("Type", style="blue")
        table.add_column("Oracle Text")
        table.add_column("Reason", style="dim white")

        for i, scored_card in enumerate(result.scored_cards, 1):
            card = scored_card.card
            score = scored_card.score

            mana_cost = card.mana_cost or "N/A"
            card_type = card.type_line or "Unknown"
            oracle_text = getattr(card, "oracle_text", None) or "—"
            reason = scored_card.reasoning or ""

            # Clickable name if URL available
            url = getattr(card, "scryfall_uri", None)
            name_cell = Text(card.name, style=f"link {url}") if url else Text(card.name)

            table.add_row(
                str(i),
                f"{score}/10",
                name_cell,
                mana_cost,
                card_type,
                oracle_text,
                reason,
            )

        return table

    def show_search_completion_summary(self, original_query: str, result):
        """Display a persistent summary of the completed search"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Create completion summary panel
        summary_lines = []
        summary_lines.append(f"[bold]Search Query:[/bold] {original_query}")
        
        if hasattr(self, 'current_query') and self.current_query:
            summary_lines.append(f"[bold]Final Scryfall Query:[/bold] {self.current_query}")
        
        summary_lines.append(f"[bold]Iterations:[/bold] {self.current_iteration}/{self.max_iterations}")
        summary_lines.append(f"[bold]Duration:[/bold] {elapsed:.1f} seconds")
        
        if hasattr(self, 'best_score') and self.best_score > 0:
            summary_lines.append(f"[bold]Best Score:[/bold] {self.best_score:.1f}/10")
        
        if hasattr(self, 'cards_found') and self.cards_found > 0:
            summary_lines.append(f"[bold]Cards Found:[/bold] {self.cards_found}")
        
        # Keep the summary concise (avoid repeating progress history)
        
        summary = Panel(
            "\n".join(summary_lines),
            title="[green]Search Completed[/green]",
            border_style="green",
            padding=(1, 2)
        )
        console.print()
        console.print(summary)

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
                
                # Wait for any final status updates to complete
                import asyncio
                await asyncio.sleep(0.8)
                
                # Clear any remaining status
                self._clear_status()
                
                # Show completion summary
                self.show_search_completion_summary(query, result)
                
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
            
            # Wait for any final status updates to complete
            import asyncio
            await asyncio.sleep(0.8)
            
            # Clear any remaining status
            self._clear_status()
            
            # Show persistent search summary
            self.show_search_completion_summary(query, result)
            
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