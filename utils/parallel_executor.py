"""
Parallel execution utilities for running agent trials concurrently
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any
import time


class ParallelTrialExecutor:
    """Execute multiple agent trials in parallel"""

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel executor

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers

    def run_trials_parallel(self, agent_func: Callable, trial_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run multiple trials in parallel

        Args:
            agent_func: Function to execute for each trial (should accept config dict)
            trial_configs: List of configuration dicts for each trial

        Returns:
            List of results from each trial
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all trials
            future_to_config = {
                executor.submit(self._execute_trial, agent_func, config): config
                for config in trial_configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Trial failed: {e}")
                    results.append({
                        'error': str(e),
                        'config': config
                    })

        return results

    def _execute_trial(self, func: Callable, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trial with timing"""
        start_time = time.time()

        try:
            result = func(**config)
            result['execution_time'] = time.time() - start_time
            return result
        except Exception as e:
            return {
                'error': str(e),
                'config': config,
                'execution_time': time.time() - start_time
            }


class BatchExecutor:
    """Execute batch operations in parallel"""

    def __init__(self, max_workers: int = 4):
        """Initialize batch executor"""
        self.max_workers = max_workers

    def execute_batch(self, func: Callable, items: List[Any],
                     show_progress: bool = True) -> List[Any]:
        """
        Execute function on batch of items in parallel

        Args:
            func: Function to execute on each item
            items: List of items to process
            show_progress: Whether to print progress

        Returns:
            List of results
        """
        results = []
        total = len(items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(func, item): (i, item)
                for i, item in enumerate(items)
            }

            completed = 0
            for future in as_completed(future_to_item):
                idx, item = future_to_item[future]
                try:
                    result = future.result()
                    results.append((idx, result))

                    completed += 1
                    if show_progress and completed % max(1, total // 10) == 0:
                        print(f"Progress: {completed}/{total} ({completed/total*100:.0f}%)")

                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    results.append((idx, None))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


def run_dual_trials(agent_func: Callable, trial_1_config: Dict, trial_2_config: Dict,
                   run_parallel: bool = True) -> tuple:
    """
    Run two trials (e.g., Trial 1 and Trial 2 of Reflexion)

    Args:
        agent_func: Agent function to run
        trial_1_config: Config for first trial
        trial_2_config: Config for second trial
        run_parallel: Whether to run in parallel (default True)

    Returns:
        Tuple of (trial_1_result, trial_2_result)
    """
    if run_parallel:
        executor = ParallelTrialExecutor(max_workers=2)
        results = executor.run_trials_parallel(agent_func, [trial_1_config, trial_2_config])
        return results[0], results[1]
    else:
        # Sequential execution
        trial_1_result = agent_func(**trial_1_config)
        trial_2_result = agent_func(**trial_2_config)
        return trial_1_result, trial_2_result
