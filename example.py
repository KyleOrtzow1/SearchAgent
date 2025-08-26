import asyncio
import os
from dotenv import load_dotenv
from orchestrator import SearchOrchestrator

# Load environment variables from .env file
load_dotenv()

async def main():
    """Example usage of the MTG Search Agent"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize the orchestrator
    tags_file_path = "tags.json"
    
    # Ask user about streaming preference
    print("MTG Card Search Agent")
    print("====================")
    print()
    print("Would you like to enable real-time streaming?")
    print("1. Yes - See AI thinking in real-time (uses more tokens)")
    print("2. No - Optimized performance mode (default)")
    
    while True:
        choice = input("\nChoice (1/2 or Enter for default): ").strip()
        if choice in ['1', 'yes', 'y']:
            enable_streaming = True
            print("🌊 Streaming enabled - you'll see real-time token generation")
            break
        elif choice in ['2', 'no', 'n', '']:
            enable_streaming = False
            print("⚡ Streaming disabled - optimized for performance and token usage")
            break
        else:
            print("Please enter 1 or 2 (or press Enter for default)")
    
    orchestrator = SearchOrchestrator(tags_file_path, enable_streaming=enable_streaming)
    
    # Example searches
    example_requests = [
        "I want a cheap red creature with haste that can deal damage quickly",
        "Find me a blue instant that can counter spells and draw cards", 
        "I need a green ramp spell that puts lands into play",
        "Show me artifacts that cost 2 mana and provide mana acceleration",
        "Find legendary creatures that are good commanders for a tokens deck"
    ]
    
    print()
    
    while True:
        print("Example requests:")
        for i, req in enumerate(example_requests, 1):
            print(f"{i}. {req}")
        print(f"{len(example_requests) + 1}. Enter custom request")
        print("0. Exit")
        print()
        
        try:
            choice = input("Select an option (0-6): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == str(len(example_requests) + 1):
                request = input("\nEnter your card search request: ").strip()
                if not request:
                    print("Empty request. Please try again.")
                    continue
            elif choice.isdigit() and 1 <= int(choice) <= len(example_requests):
                request = example_requests[int(choice) - 1]
                print(f"\nSearching for: {request}")
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Perform the search
            print("\nStarting search...")
            result = await orchestrator.search(request)
            
            # Display results
            orchestrator.print_final_results(result)
            
            # Ask if user wants to continue
            continue_choice = input("\nPress Enter to continue or 'q' to quit: ").strip().lower()
            if continue_choice == 'q':
                print("Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())