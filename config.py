"""
Configuration file for Reflexion Business Intelligence Agent
"""
import os

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables or defaults

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o-mini"  # Valid models: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
TEMPERATURE = 0.7

# Agent Configuration
MAX_ITERATIONS = 15  # Increased to allow agent to complete reasoning
AGENT_VERBOSE = True

# Evaluation Configuration
SUCCESS_THRESHOLD = 10.0  # MAPE threshold for success (10%)

# Memory Configuration
MAX_MEMORY_ITEMS = 100
MEMORY_RETRIEVAL_TOP_K = 3
USE_VECTOR_MEMORY = True  # Use FAISS vector memory (True) or keyword memory (False)
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model

# Caching Configuration
ENABLE_CACHING = True
CACHE_TTL_HOURS = 24  # Time-to-live for cache entries

# Parallel Execution Configuration
USE_PARALLEL_TRIALS = True  # Run trials in parallel for faster responses
MAX_WORKERS = 4  # Maximum concurrent workers

# Smart Tools Configuration
USE_SMART_TOOLS = True  # Use enhanced tools with statistical analysis

# Dataset Configuration
DATASET_PATH = "data/walmart_sales.csv"
KAGGLE_DATASET = "yasserh/walmart-dataset"

# Test Configuration
NUM_TEST_CASES = 10
MIN_HISTORICAL_WEEKS = 8  # Minimum weeks needed for prediction
