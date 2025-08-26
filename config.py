import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCRYFALL_BASE_URL = "https://api.scryfall.com"
SCRYFALL_RATE_LIMIT_MS = 100  # 100ms between requests
MAX_SEARCH_LOOPS = 5
MAX_RESULTS_PER_SEARCH = 500

# Parallel evaluation settings
ENABLE_PARALLEL_EVALUATION = True
EVALUATION_BATCH_SIZE = 10  # Cards per batch for parallel processing

# Results display settings
TOP_CARDS_TO_DISPLAY = 15  # Number of highest scoring cards to show in final results

# Search continuation settings
STOP_LOOP_CONFIDENCE_THRESHOLD = 6  # Minimum average score to stop search loop (1-10 scale)