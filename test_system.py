"""
Test script to verify system functionality
"""
import sys
import traceback

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import config
        print("✓ config imported")

        from utils.data_loader import WalmartDataLoader
        print("✓ WalmartDataLoader imported")

        from utils.evaluator import PredictionEvaluator
        print("✓ PredictionEvaluator imported")

        from utils.prompts import REACT_SYSTEM_PROMPT
        print("✓ prompts imported")

        from agents.tools import SalesAnalysisTools
        print("✓ SalesAnalysisTools imported")

        from agents.baseline_agent import BaselineAgent
        print("✓ BaselineAgent imported")

        from memory.episodic_memory import EpisodicMemory
        print("✓ EpisodicMemory imported")

        from agents.reflexion_agent import ReflexionAgent
        print("✓ ReflexionAgent imported")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader functionality"""
    print("\nTesting data loader...")
    try:
        from utils.data_loader import WalmartDataLoader

        loader = WalmartDataLoader()
        df = loader.load_data(use_sample=True)

        print(f"✓ Sample data loaded: {len(df)} rows")

        test_cases = loader.generate_test_cases(num_cases=3)
        print(f"✓ Generated {len(test_cases)} test cases")

        stats = loader.get_summary_stats()
        print(f"✓ Statistics: {stats['num_stores']} stores, {stats['num_departments']} departments")

        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluator():
    """Test evaluator functionality"""
    print("\nTesting evaluator...")
    try:
        from utils.evaluator import PredictionEvaluator

        evaluator = PredictionEvaluator()

        # Test successful prediction
        result1 = evaluator.evaluate(prediction=10000, actual=10500)
        print(f"✓ Evaluation 1: MAPE={result1['mape']}%, Success={result1['success']}")

        # Test failed prediction
        result2 = evaluator.evaluate(prediction=10000, actual=15000)
        print(f"✓ Evaluation 2: MAPE={result2['mape']}%, Success={result2['success']}")

        # Test improvement calculation
        improvement = evaluator.calculate_improvement(result2, result1)
        print(f"✓ Improvement calculation: {improvement['mape_improvement']}%")

        return True
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        traceback.print_exc()
        return False


def test_memory():
    """Test episodic memory"""
    print("\nTesting episodic memory...")
    try:
        from memory.episodic_memory import EpisodicMemory

        memory = EpisodicMemory(use_embeddings=False)

        # Add some memories
        memory.add_memory(
            task_description="Predict Store 1, Dept 1",
            prediction=10000,
            actual=12000,
            reflection="Need to check for seasonality",
            success=False,
            metadata={'store': 1, 'department': 1}
        )

        memory.add_memory(
            task_description="Predict Store 2, Dept 2",
            prediction=15000,
            actual=15500,
            reflection="Good prediction using trend analysis",
            success=True,
            metadata={'store': 2, 'department': 2}
        )

        print(f"✓ Added 2 memories")

        # Retrieve memories
        relevant = memory.retrieve_relevant_memories("Predict Store 1, Dept 3")
        print(f"✓ Retrieved {len(relevant)} relevant memories")

        # Get summary
        summary = memory.get_memory_summary()
        print(f"✓ Memory summary: {summary['total_memories']} total, {summary['success_rate']:.1f}% success rate")

        return True
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        traceback.print_exc()
        return False


def test_tools():
    """Test agent tools"""
    print("\nTesting agent tools...")
    try:
        from utils.data_loader import WalmartDataLoader
        from agents.tools import SalesAnalysisTools
        import pandas as pd

        # Create sample historical data
        loader = WalmartDataLoader()
        df = loader.load_data(use_sample=True)
        historical_data = df.head(20)

        # Create tools
        tools_creator = SalesAnalysisTools(historical_data)
        tools = tools_creator.get_tools()

        print(f"✓ Created {len(tools)} tools")
        print(f"  Tools: {[t.name for t in tools]}")

        return True
    except Exception as e:
        print(f"✗ Tools test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("REFLEXION AGENT SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Evaluator", test_evaluator),
        ("Memory", test_memory),
        ("Tools", test_tools),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:20s} {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
