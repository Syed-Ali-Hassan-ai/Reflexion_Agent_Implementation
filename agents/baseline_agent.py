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

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

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

        except Exception as e:
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
        output = result.get('output', '')

        # Look for the make_prediction tool call
        for step in result.get('intermediate_steps', []):
            action, observation = step
            if action.tool == 'make_prediction':
                try:
                    # Extract the predicted_sales argument
                    prediction = float(action.tool_input.get('predicted_sales', 0))
                    return prediction
                except:
                    pass

        # Fallback: try to extract from output text
        # Look for patterns like $XX,XXX.XX or numbers
        numbers = re.findall(r'\$?([0-9,]+\.?[0-9]*)', output)
        if numbers:
            try:
                # Clean and convert the first number found
                cleaned = numbers[0].replace(',', '')
                return float(cleaned)
            except:
                pass

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
