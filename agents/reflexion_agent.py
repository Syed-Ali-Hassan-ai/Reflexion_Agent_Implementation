"""
Reflexion agent with self-reflection and episodic memory
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import re
import config
from agents.tools import SalesAnalysisTools
from memory.episodic_memory import EpisodicMemory
from utils.prompts import (
    REFLEXION_SYSTEM_PROMPT,
    MEMORY_CONTEXT_TEMPLATE,
    REFLECTION_PROMPT,
    TASK_DESCRIPTION_TEMPLATE
)
from utils.evaluator import PredictionEvaluator


class ReflexionAgent:
    """Reflexion agent that learns from mistakes through self-reflection"""

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the Reflexion agent

        Args:
            api_key: OpenAI API key
            model: Model name (default from config)
        """
        self.api_key = api_key
        self.model = model or config.DEFAULT_MODEL
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=self.model,
            temperature=config.TEMPERATURE
        )
        self.memory = EpisodicMemory(use_embeddings=False)  # Simple keyword matching
        self.evaluator = PredictionEvaluator()
        self.agent_executor = None
        self.execution_log = []

    def setup_agent(self, historical_data, task_description: str = ""):
        """
        Setup the agent with tools and memory for the given data

        Args:
            historical_data: DataFrame with historical sales data
            task_description: Description of current task (for memory retrieval)
        """
        # Create tools
        tools_creator = SalesAnalysisTools(historical_data)
        tools = tools_creator.get_tools()

        # Retrieve relevant memories
        relevant_memories = []
        if task_description:
            relevant_memories = self.memory.retrieve_relevant_memories(task_description)

        # Format memory context
        memory_context = ""
        if relevant_memories:
            formatted_memories = self.memory.format_memories_for_prompt(relevant_memories)
            memory_context = MEMORY_CONTEXT_TEMPLATE.format(reflections=formatted_memories)

        # Create prompt template with memory
        system_prompt = REFLEXION_SYSTEM_PROMPT.format(memory_context=memory_context)

        template = f"""{system_prompt}

You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )

        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=config.AGENT_VERBOSE,
            max_iterations=config.MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def predict(self, test_case: Dict[str, Any], trial_number: int = 1) -> Dict[str, Any]:
        """
        Make a prediction for a test case

        Args:
            test_case: Test case dictionary with store, department, and historical data
            trial_number: Trial number (for logging)

        Returns:
            Dictionary with prediction and execution details
        """
        # Create task description
        task = TASK_DESCRIPTION_TEMPLATE.format(
            store=test_case['store'],
            department=test_case['department'],
            week=test_case['test_week']
        )

        # Setup agent with historical data and memory
        self.setup_agent(test_case['historical_data'], task)

        # Execute agent
        try:
            result = self.agent_executor.invoke({"input": task})

            # Extract prediction from the final answer
            prediction = self._extract_prediction(result)

            # Store execution log
            self.execution_log = result.get('intermediate_steps', [])

            return {
                'prediction': prediction,
                'reasoning': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', []),
                'success': True,
                'trial': trial_number,
                'task_description': task
            }

        except StopIteration as e:
            # Handle StopIteration specifically (common LangChain issue)
            import traceback
            print(f"StopIteration error caught. This is often caused by tool execution issues.")
            print(f"Traceback: {traceback.format_exc()}")

            # Try to extract any partial results
            partial_prediction = 0.0
            if self.execution_log:
                # Attempt to extract from partial execution
                partial_result = {'intermediate_steps': self.execution_log, 'output': ''}
                partial_prediction = self._extract_prediction(partial_result)

            return {
                'prediction': partial_prediction,
                'reasoning': f"Agent execution incomplete (StopIteration). This may be due to tool parameter issues. Partial prediction extracted: ${partial_prediction:,.2f}",
                'intermediate_steps': self.execution_log,
                'success': partial_prediction > 0,
                'error': f"StopIteration: {str(e)}",
                'trial': trial_number,
                'task_description': task
            }

        except Exception as e:
            import traceback
            print(f"Error during agent execution: {e}")
            print(f"Traceback: {traceback.format_exc()}")

            return {
                'prediction': 0.0,
                'reasoning': f"Error during prediction: {str(e)}",
                'intermediate_steps': [],
                'success': False,
                'error': str(e),
                'trial': trial_number,
                'task_description': task
            }

    def generate_reflection(self, task_description: str, prediction: float,
                          actual: float, error_percentage: float) -> str:
        """
        Generate a reflection on why the prediction failed

        Args:
            task_description: Description of the task
            prediction: Predicted value
            actual: Actual value
            error_percentage: Percentage error

        Returns:
            Reflection text
        """
        reflection_prompt = REFLECTION_PROMPT.format(
            task_description=task_description,
            prediction=prediction,
            actual=actual,
            error_percentage=error_percentage
        )

        try:
            response = self.llm.invoke(reflection_prompt)
            reflection = response.content.strip()
            return reflection
        except Exception as e:
            return f"Unable to generate reflection: {str(e)}"

    def run_trial_with_reflection(self, test_case: Dict[str, Any],
                                  num_trials: int = 2) -> List[Dict[str, Any]]:
        """
        Run multiple trials with reflection between trials

        Args:
            test_case: Test case to evaluate
            num_trials: Number of trials to run

        Returns:
            List of trial results
        """
        results = []

        for trial in range(1, num_trials + 1):
            # Make prediction
            prediction_result = self.predict(test_case, trial_number=trial)

            if not prediction_result['success']:
                results.append(prediction_result)
                continue

            # Evaluate prediction
            evaluation = self.evaluator.evaluate(
                prediction_result['prediction'],
                test_case['actual_sales']
            )

            # Combine results
            trial_result = {
                **prediction_result,
                'evaluation': evaluation,
                'reflection': None
            }

            results.append(trial_result)

            # Generate reflection if failed and not the last trial
            if not evaluation['success'] and trial < num_trials:
                reflection = self.generate_reflection(
                    task_description=prediction_result['task_description'],
                    prediction=prediction_result['prediction'],
                    actual=test_case['actual_sales'],
                    error_percentage=evaluation['percentage_error']
                )

                trial_result['reflection'] = reflection

                # Store in memory
                self.memory.add_memory(
                    task_description=prediction_result['task_description'],
                    prediction=prediction_result['prediction'],
                    actual=test_case['actual_sales'],
                    reflection=reflection,
                    success=evaluation['success'],
                    metadata={
                        'store': test_case['store'],
                        'department': test_case['department']
                    }
                )

            # If successful, still store as positive example
            elif evaluation['success']:
                success_insight = "This prediction was successful. The approach used was effective."
                self.memory.add_memory(
                    task_description=prediction_result['task_description'],
                    prediction=prediction_result['prediction'],
                    actual=test_case['actual_sales'],
                    reflection=success_insight,
                    success=True,
                    metadata={
                        'store': test_case['store'],
                        'department': test_case['department']
                    }
                )

        return results

    def _extract_prediction(self, result: Dict[str, Any]) -> float:
        """
        Extract numerical prediction from agent output

        Args:
            result: Agent execution result

        Returns:
            Predicted sales value
        """
        import json

        output = result.get('output', '')

        # Look for the make_prediction tool call
        for step in result.get('intermediate_steps', []):
            action, observation = step
            if action.tool == 'make_prediction':
                try:
                    tool_input = action.tool_input

                    # Handle different formats of tool_input
                    if isinstance(tool_input, dict):
                        # Direct dictionary access
                        prediction = float(tool_input.get('predicted_sales', 0))
                        if prediction > 0:
                            return prediction
                    elif isinstance(tool_input, str):
                        # Try parsing as JSON
                        try:
                            parsed = json.loads(tool_input)
                            if isinstance(parsed, dict):
                                prediction = float(parsed.get('predicted_sales', 0))
                                if prediction > 0:
                                    return prediction
                        except:
                            # Try extracting number from string directly
                            nums = re.findall(r'predicted_sales["\']?\s*[:=]\s*([0-9.]+)', tool_input)
                            if nums:
                                return float(nums[0])

                    # Try to extract from observation (the tool's response)
                    if observation and 'Prediction recorded:' in observation:
                        nums = re.findall(r'\$([0-9,]+\.?[0-9]*)', observation)
                        if nums:
                            cleaned = nums[0].replace(',', '')
                            return float(cleaned)

                except Exception as e:
                    print(f"Debug: Error extracting from tool call: {e}")
                    print(f"Debug: tool_input type: {type(action.tool_input)}")
                    print(f"Debug: tool_input value: {action.tool_input}")

        # Fallback 1: Look in all observations for recorded predictions
        for step in result.get('intermediate_steps', []):
            action, observation = step
            if observation and 'Prediction recorded:' in str(observation):
                nums = re.findall(r'\$([0-9,]+\.?[0-9]*)', str(observation))
                if nums:
                    try:
                        cleaned = nums[0].replace(',', '')
                        return float(cleaned)
                    except:
                        pass

        # Fallback 2: try to extract from output text
        # Look for patterns like $XX,XXX.XX or numbers
        numbers = re.findall(r'\$([0-9,]+\.?[0-9]*)', output)
        if numbers:
            try:
                # Clean and convert the first large number found
                cleaned = numbers[0].replace(',', '')
                value = float(cleaned)
                if value > 100:  # Sanity check - sales should be at least $100
                    return value
            except:
                pass

        # Fallback 3: Look for any large number in output
        numbers = re.findall(r'([0-9,]+\.?[0-9]+)', output)
        for num_str in numbers:
            try:
                cleaned = num_str.replace(',', '')
                value = float(cleaned)
                if value > 1000:  # Likely a sales figure
                    return value
            except:
                pass

        print(f"Warning: Could not extract prediction from agent output")
        print(f"Output: {output[:200]}")
        return 0.0

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """
        Get formatted execution trace

        Returns:
            List of execution steps
        """
        trace = []
        for i, (action, observation) in enumerate(self.execution_log):
            trace.append({
                'step': i + 1,
                'action': action.tool,
                'input': action.tool_input,
                'observation': observation
            })
        return trace

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of stored memories"""
        return self.memory.get_memory_summary()

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories"""
        return self.memory.get_all_memories()

    def clear_memory(self):
        """Clear all memories"""
        self.memory.clear_memories()
