"""
Baseline ReAct agent for sales forecasting
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List
import re
import config
from agents.tools import SalesAnalysisTools
from utils.prompts import REACT_SYSTEM_PROMPT, TASK_DESCRIPTION_TEMPLATE


class BaselineAgent:
    """Standard ReAct agent for sales forecasting"""

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the baseline agent

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
        self.agent_executor = None
        self.execution_log = []

    def setup_agent(self, historical_data):
        """
        Setup the agent with tools for the given data

        Args:
            historical_data: DataFrame with historical sales data
        """
        # Create tools
        tools_creator = SalesAnalysisTools(historical_data)
        tools = tools_creator.get_tools()

        # Create prompt template for ReAct
        template = """You are a sales forecasting expert. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
- After writing "Action Input:", STOP immediately. Do NOT write anything else.
- NEVER write "Observation:" yourself - the system will provide it automatically
- After seeing an Observation, start with "Thought:" for your next step
- Only write "Final Answer:" when you are completely done and have all information

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

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

    def predict(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a test case

        Args:
            test_case: Test case dictionary with store, department, and historical data

        Returns:
            Dictionary with prediction and execution details
        """
        # Setup agent with historical data
        self.setup_agent(test_case['historical_data'])

        # Create task description
        task = TASK_DESCRIPTION_TEMPLATE.format(
            store=test_case['store'],
            department=test_case['department'],
            week=test_case['test_week']
        )

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
                'success': True
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
                'error': f"StopIteration: {str(e)}"
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
                'error': str(e)
            }

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
