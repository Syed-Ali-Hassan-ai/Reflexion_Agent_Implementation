"""
Evaluator for sales predictions
"""
from typing import Dict, Tuple
import config


class PredictionEvaluator:
    """Evaluates agent predictions against ground truth"""

    @staticmethod
    def evaluate(prediction: float, actual: float) -> Dict[str, any]:
        """
        Evaluate a prediction against actual value

        Args:
            prediction: Predicted sales value
            actual: Actual sales value

        Returns:
            Dictionary with evaluation metrics
        """
        # Handle edge cases
        if actual == 0:
            if prediction == 0:
                mape = 0.0
                error = 0.0
            else:
                mape = 100.0
                error = abs(prediction)
        else:
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = abs(actual - prediction) / abs(actual) * 100
            error = abs(actual - prediction)

        # Calculate percentage error
        percentage_error = ((prediction - actual) / actual * 100) if actual != 0 else 0

        # Determine success
        success = mape < config.SUCCESS_THRESHOLD

        return {
            'prediction': prediction,
            'actual': actual,
            'mape': round(mape, 2),
            'absolute_error': round(error, 2),
            'percentage_error': round(percentage_error, 2),
            'success': success,
            'within_threshold': f"±{config.SUCCESS_THRESHOLD}%"
        }

    @staticmethod
    def calculate_improvement(trial1_result: Dict, trial2_result: Dict) -> Dict[str, any]:
        """
        Calculate improvement between two trials

        Args:
            trial1_result: Evaluation result from trial 1
            trial2_result: Evaluation result from trial 2

        Returns:
            Dictionary with improvement metrics
        """
        mape_improvement = trial1_result['mape'] - trial2_result['mape']
        improvement_percentage = (mape_improvement / trial1_result['mape'] * 100) if trial1_result['mape'] != 0 else 0

        return {
            'mape_improvement': round(mape_improvement, 2),
            'improvement_percentage': round(improvement_percentage, 2),
            'trial1_success': trial1_result['success'],
            'trial2_success': trial2_result['success'],
            'became_successful': not trial1_result['success'] and trial2_result['success']
        }

    @staticmethod
    def aggregate_results(results: list) -> Dict[str, any]:
        """
        Aggregate multiple evaluation results

        Args:
            results: List of evaluation result dictionaries

        Returns:
            Aggregated statistics
        """
        if not results:
            return {}

        total_mape = sum(r['mape'] for r in results)
        total_success = sum(1 for r in results if r['success'])

        return {
            'total_cases': len(results),
            'average_mape': round(total_mape / len(results), 2),
            'success_count': total_success,
            'success_rate': round(total_success / len(results) * 100, 2),
            'best_mape': round(min(r['mape'] for r in results), 2),
            'worst_mape': round(max(r['mape'] for r in results), 2)
        }
