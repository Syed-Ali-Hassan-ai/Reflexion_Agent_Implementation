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
DEFAULT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# Agent Configuration
MAX_ITERATIONS = 5
AGENT_VERBOSE = True

# Evaluation Configuration
SUCCESS_THRESHOLD = 10.0  # MAPE threshold for success (10%)

# Memory Configuration
MAX_MEMORY_ITEMS = 100
MEMORY_RETRIEVAL_TOP_K = 3

# Dataset Configuration
DATASET_PATH = "data/walmart_sales.csv"
KAGGLE_DATASET = "yasserh/walmart-dataset"

# Test Configuration
NUM_TEST_CASES = 10
MIN_HISTORICAL_WEEKS = 8  # Minimum weeks needed for prediction
