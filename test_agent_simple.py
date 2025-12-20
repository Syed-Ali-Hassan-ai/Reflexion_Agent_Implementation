"""
Simple diagnostic test for the agents
Run this to see if agents are working and what they're doing
"""
import os
import pandas as pd
from agents.baseline_agent import BaselineAgent
from utils.data_loader import WalmartDataLoader
from utils.evaluator import PredictionEvaluator

def test_agent():
    print("=" * 60)
    print("AGENT DIAGNOSTIC TEST")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("Or enter it now:")
        api_key = input("API Key: ").strip()

    if not api_key:
        print("Cannot proceed without API key")
        return

    print(f"\n✓ API Key found: {api_key[:8]}...{api_key[-4:]}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    loader = WalmartDataLoader()
    df = loader.load_data(use_sample=True)
    print(f"✓ Data loaded: {len(df)} rows")

    # Generate test case
    print("\n" + "=" * 60)
    print("Generating test case...")
    test_cases = loader.generate_test_cases(num_cases=1)
    if not test_cases:
        print("❌ Failed to generate test cases")
        return

    test_case = test_cases[0]
    print(f"✓ Test case: {test_case['description']}")
    print(f"  Store: {test_case['store']}")
    print(f"  Department: {test_case['department']}")
    print(f"  Actual Sales: ${test_case['actual_sales']:,.2f}")
    print(f"  Historical weeks: {len(test_case['historical_data'])}")

    # Test agent
    print("\n" + "=" * 60)
    print("Testing Baseline Agent...")
    print("This will call OpenAI API (takes 30-60 seconds)...")

    try:
        agent = BaselineAgent(api_key)
        result = agent.predict(test_case)

        print("\n" + "=" * 60)
        print("AGENT RESULT:")
        print("=" * 60)

        if result['success']:
            print(f"✓ Agent executed successfully")
            print(f"\nPrediction: ${result['prediction']:,.2f}")
            print(f"Actual:     ${test_case['actual_sales']:,.2f}")

            # Evaluate
            evaluator = PredictionEvaluator()
            eval_result = evaluator.evaluate(result['prediction'], test_case['actual_sales'])

            print(f"\nError: {eval_result['mape']:.2f}%")
            print(f"Success: {'✓ YES' if eval_result['success'] else '✗ NO'} (threshold: ±10%)")

            # Show reasoning trace
            print("\n" + "=" * 60)
            print("AGENT REASONING TRACE:")
            print("=" * 60)

            trace = agent.get_execution_trace()
            if trace:
                for step in trace:
                    print(f"\nStep {step['step']}: {step['action']}")
                    print(f"  Input: {step['input']}")
                    print(f"  Output: {step['observation'][:200]}...")
            else:
                print("No trace available")

            # Show full output
            print("\n" + "=" * 60)
            print("FINAL ANSWER:")
            print("=" * 60)
            print(result['reasoning'])

        else:
            print(f"❌ Agent failed")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_agent()
