"""
Streamlit app for Reflexion Business Intelligence Agent
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Import project modules
from utils.data_loader import WalmartDataLoader
from utils.evaluator import PredictionEvaluator
from agents.baseline_agent import BaselineAgent
from agents.reflexion_agent import ReflexionAgent
import config

# Page configuration
st.set_page_config(
    page_title="Reflexion Business Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .failure-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = WalmartDataLoader()
        st.session_state.data_loaded = False
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = []
    if 'current_test_case' not in st.session_state:
        st.session_state.current_test_case = None
    if 'baseline_agent' not in st.session_state:
        st.session_state.baseline_agent = None
    if 'reflexion_agent' not in st.session_state:
        st.session_state.reflexion_agent = None
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None


init_session_state()


# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key to use the agents"
)

if api_key:
    config.OPENAI_API_KEY = api_key

st.sidebar.markdown("---")

agent_type = st.sidebar.radio(
    "Select Agent Type",
    ["Baseline ReAct", "Reflexion Agent"],
    help="Choose between standard ReAct agent and Reflexion agent"
)

num_trials = st.sidebar.slider(
    "Number of Trials",
    min_value=1,
    max_value=3,
    value=2,
    help="Number of prediction trials (for Reflexion agent)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This demo showcases **Reflexion** - an approach that enables agents to
    learn from their mistakes through self-reflection and episodic memory.

    **Key Features:**
    - Self-reflection after failures
    - Episodic memory storage
    - Iterative improvement
    - Comparison with baseline
    """
)


# Main app
st.markdown('<p class="main-header">🧠 Reflexion Business Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown("**Self-Improving AI for Sales Forecasting**")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Demo",
    "📈 Batch Comparison",
    "🧠 Memory Explorer",
    "📖 How It Works"
])

# Tab 1: Live Demo
with tab1:
    st.markdown("## Live Sales Forecasting Demo")

    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading Walmart sales dataset..."):
            st.session_state.data_loader.load_data(use_sample=True)
            st.session_state.test_cases = st.session_state.data_loader.generate_test_cases(
                num_cases=config.NUM_TEST_CASES
            )
            st.session_state.data_loaded = True

    # Show dataset overview
    with st.expander("📊 Dataset Overview", expanded=False):
        st.write("**Sample of Historical Sales Data:**")
        df = st.session_state.data_loader.df
        st.dataframe(df.head(20), use_container_width=True)

        stats = st.session_state.data_loader.get_summary_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{stats['total_records']:,}")
        col2.metric("Stores", stats['num_stores'])
        col3.metric("Departments", stats['num_departments'])
        col4.metric("Avg Sales", f"${stats['avg_sales']:,.2f}")

    st.markdown("---")

    # Generate test case
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🎲 Generate Random Test Case", type="primary"):
            if st.session_state.test_cases:
                st.session_state.current_test_case = np.random.choice(st.session_state.test_cases)

    with col2:
        if st.session_state.current_test_case:
            tc = st.session_state.current_test_case
            st.info(f"**Test Case:** {tc['description']}")

    # Show test case details
    if st.session_state.current_test_case:
        tc = st.session_state.current_test_case

        st.markdown("### 📋 Test Case Details")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Store", tc['store'])
        col2.metric("Department", tc['department'])
        col3.metric("Actual Sales", f"${tc['actual_sales']:,.2f}")
        col4.metric("Historical Weeks", len(tc['historical_data']))

        # Show historical data chart
        with st.expander("📈 Historical Sales Trend", expanded=True):
            hist_df = tc['historical_data'].copy()
            fig = px.line(
                hist_df,
                x=hist_df.index,
                y=[col for col in hist_df.columns if 'sales' in col.lower()][0],
                title="Historical Weekly Sales",
                labels={'x': 'Week', 'y': 'Sales ($)'}
            )
            fig.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Run agent
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to run the agent.")
        else:
            if st.button("🚀 Run Agent", type="primary"):
                with st.spinner(f"Running {agent_type}..."):
                    try:
                        if agent_type == "Baseline ReAct":
                            # Initialize baseline agent
                            if st.session_state.baseline_agent is None:
                                st.session_state.baseline_agent = BaselineAgent(api_key)

                            agent = st.session_state.baseline_agent
                            result = agent.predict(tc)

                            # Evaluate
                            evaluator = PredictionEvaluator()
                            evaluation = evaluator.evaluate(result['prediction'], tc['actual_sales'])

                            # Display results
                            st.markdown("### 🎯 Prediction Results")

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Prediction", f"${result['prediction']:,.2f}")
                            col2.metric("Actual", f"${tc['actual_sales']:,.2f}")
                            col3.metric("Error", f"{evaluation['mape']:.2f}%")

                            if evaluation['success']:
                                st.markdown('<div class="success-box">✅ <b>SUCCESS!</b> Prediction within ±10% threshold.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="failure-box">❌ <b>FAILED</b> Prediction exceeded ±10% threshold.</div>', unsafe_allow_html=True)

                            # Show reasoning trace
                            with st.expander("🔍 Agent Reasoning Trace", expanded=False):
                                trace = agent.get_execution_trace()
                                for step in trace:
                                    st.markdown(f"**Step {step['step']}: {step['action']}**")
                                    st.write(f"Input: {step['input']}")
                                    st.write(f"Observation: {step['observation']}")
                                    st.markdown("---")

                        else:  # Reflexion Agent
                            # Initialize reflexion agent
                            if st.session_state.reflexion_agent is None:
                                st.session_state.reflexion_agent = ReflexionAgent(api_key)

                            agent = st.session_state.reflexion_agent

                            # Run trials with reflection
                            trials_results = agent.run_trial_with_reflection(tc, num_trials=num_trials)

                            st.markdown("### 🎯 Multi-Trial Results")

                            # Display each trial
                            for i, trial_result in enumerate(trials_results, 1):
                                st.markdown(f"#### Trial {i}")

                                col1, col2, col3 = st.columns(3)
                                col1.metric("Prediction", f"${trial_result['prediction']:,.2f}")
                                col2.metric("Actual", f"${tc['actual_sales']:,.2f}")
                                col3.metric("Error", f"{trial_result['evaluation']['mape']:.2f}%")

                                if trial_result['evaluation']['success']:
                                    st.markdown('<div class="success-box">✅ <b>SUCCESS!</b> Prediction within ±10% threshold.</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="failure-box">❌ <b>FAILED</b> Prediction exceeded ±10% threshold.</div>', unsafe_allow_html=True)

                                # Show reflection if generated
                                if trial_result.get('reflection'):
                                    with st.expander("💭 Self-Reflection", expanded=True):
                                        st.markdown(f"**Reflection:**")
                                        st.write(trial_result['reflection'])

                                # Show reasoning trace
                                with st.expander("🔍 Agent Reasoning Trace", expanded=False):
                                    trace = agent.get_execution_trace()
                                    for step in trace:
                                        st.markdown(f"**Step {step['step']}: {step['action']}**")
                                        st.write(f"Input: {step['input']}")
                                        st.write(f"Observation: {step['observation']}")
                                        st.markdown("---")

                                st.markdown("---")

                            # Show improvement if multiple trials
                            if len(trials_results) > 1:
                                st.markdown("### 📊 Improvement Analysis")
                                evaluator = PredictionEvaluator()
                                improvement = evaluator.calculate_improvement(
                                    trials_results[0]['evaluation'],
                                    trials_results[-1]['evaluation']
                                )

                                col1, col2, col3 = st.columns(3)
                                col1.metric("MAPE Improvement", f"{improvement['mape_improvement']:.2f}%")
                                col2.metric("Improvement %", f"{improvement['improvement_percentage']:.1f}%")
                                col3.metric(
                                    "Outcome",
                                    "Became Successful" if improvement['became_successful'] else "No Change"
                                )

                    except Exception as e:
                        st.error(f"Error running agent: {str(e)}")


# Tab 2: Batch Comparison
with tab2:
    st.markdown("## Batch Comparison: Baseline vs Reflexion")

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to run batch comparison.")
    else:
        if st.button("🚀 Run Batch Comparison", type="primary"):
            with st.spinner("Running batch comparison on 10 test cases..."):
                try:
                    # Initialize agents
                    baseline_agent = BaselineAgent(api_key)
                    reflexion_agent = ReflexionAgent(api_key)
                    evaluator = PredictionEvaluator()

                    results = []

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, tc in enumerate(st.session_state.test_cases[:10]):
                        status_text.text(f"Processing test case {i+1}/10...")

                        # Baseline prediction
                        baseline_result = baseline_agent.predict(tc)
                        baseline_eval = evaluator.evaluate(baseline_result['prediction'], tc['actual_sales'])

                        # Reflexion prediction (2 trials)
                        reflexion_trials = reflexion_agent.run_trial_with_reflection(tc, num_trials=2)
                        reflexion_eval = reflexion_trials[-1]['evaluation']

                        results.append({
                            'Test Case': tc['description'],
                            'Store': tc['store'],
                            'Dept': tc['department'],
                            'Actual Sales': tc['actual_sales'],
                            'Baseline Prediction': baseline_result['prediction'],
                            'Baseline Error %': baseline_eval['mape'],
                            'Baseline Success': baseline_eval['success'],
                            'Reflexion Prediction': reflexion_trials[-1]['prediction'],
                            'Reflexion Error %': reflexion_eval['mape'],
                            'Reflexion Success': reflexion_eval['success']
                        })

                        progress_bar.progress((i + 1) / 10)

                    progress_bar.empty()
                    status_text.empty()

                    # Store results
                    st.session_state.batch_results = pd.DataFrame(results)

                    st.success("✅ Batch comparison completed!")

                except Exception as e:
                    st.error(f"Error during batch comparison: {str(e)}")

    # Display results if available
    if st.session_state.batch_results is not None:
        df = st.session_state.batch_results

        st.markdown("### 📊 Detailed Results")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📈 Summary Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Baseline ReAct Agent")
            baseline_avg_error = df['Baseline Error %'].mean()
            baseline_success_rate = df['Baseline Success'].sum() / len(df) * 100

            st.metric("Average Error", f"{baseline_avg_error:.2f}%")
            st.metric("Success Rate", f"{baseline_success_rate:.1f}%")

        with col2:
            st.markdown("#### Reflexion Agent")
            reflexion_avg_error = df['Reflexion Error %'].mean()
            reflexion_success_rate = df['Reflexion Success'].sum() / len(df) * 100

            st.metric("Average Error", f"{reflexion_avg_error:.2f}%")
            st.metric("Success Rate", f"{reflexion_success_rate:.1f}%")

        # Improvement metrics
        st.markdown("### 🎯 Improvement Metrics")
        error_reduction = baseline_avg_error - reflexion_avg_error
        success_rate_improvement = reflexion_success_rate - baseline_success_rate

        col1, col2 = st.columns(2)
        col1.metric("Error Reduction", f"{error_reduction:.2f}%", delta=f"{error_reduction:.2f}%")
        col2.metric("Success Rate Improvement", f"{success_rate_improvement:.1f}%", delta=f"{success_rate_improvement:.1f}%")

        # Visualization
        st.markdown("### 📊 Visualization")

        # Bar chart comparing errors
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Baseline',
            x=df.index,
            y=df['Baseline Error %'],
            marker_color='#ff7f0e'
        ))
        fig.add_trace(go.Bar(
            name='Reflexion',
            x=df.index,
            y=df['Reflexion Error %'],
            marker_color='#2ca02c'
        ))
        fig.update_layout(
            title='Error Comparison by Test Case',
            xaxis_title='Test Case',
            yaxis_title='Error (%)',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)


# Tab 3: Memory Explorer
with tab3:
    st.markdown("## Memory Explorer")

    if st.session_state.reflexion_agent is None:
        st.info("Run the Reflexion agent first to see stored memories.")
    else:
        agent = st.session_state.reflexion_agent
        memories = agent.get_all_memories()

        if not memories:
            st.info("No memories stored yet. Run some predictions with the Reflexion agent.")
        else:
            summary = agent.get_memory_summary()

            st.markdown("### 📊 Memory Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Memories", summary['total_memories'])
            col2.metric("Successful", summary['successful_predictions'])
            col3.metric("Failed", summary['failed_predictions'])
            col4.metric("Success Rate", f"{summary['success_rate']:.1f}%")

            st.markdown("---")

            st.markdown("### 🧠 Stored Reflections")

            for i, memory in enumerate(reversed(memories), 1):
                with st.expander(f"Memory {i}: {memory['task_description'][:50]}...", expanded=False):
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", f"${memory['prediction']:,.2f}")
                    col2.metric("Actual", f"${memory['actual']:,.2f}")

                    if memory['success']:
                        st.success("✅ Successful Prediction")
                    else:
                        st.error("❌ Failed Prediction")

                    st.markdown("**Task:**")
                    st.write(memory['task_description'])

                    st.markdown("**Reflection/Insight:**")
                    st.write(memory['reflection'])

                    st.markdown("**Metadata:**")
                    st.json(memory['metadata'])

            # Search functionality
            st.markdown("---")
            st.markdown("### 🔍 Search Memories")
            search_query = st.text_input("Enter keywords to search reflections")

            if search_query:
                matching_memories = [
                    m for m in memories
                    if search_query.lower() in m['reflection'].lower() or
                       search_query.lower() in m['task_description'].lower()
                ]

                st.write(f"Found {len(matching_memories)} matching memories")

                for memory in matching_memories:
                    st.markdown(f"**{memory['task_description']}**")
                    st.write(memory['reflection'])
                    st.markdown("---")


# Tab 4: How It Works
with tab4:
    st.markdown("## How Reflexion Works")

    st.markdown("""
    ### 🎯 The Problem

    Traditional ReAct agents make predictions but **don't learn from their mistakes**.
    Each task is approached fresh, without benefiting from past experiences.

    ### 💡 The Reflexion Solution

    Reflexion adds three key components:

    1. **Self-Reflection**: After a failure, the agent analyzes WHY it failed
    2. **Episodic Memory**: Reflections are stored for future reference
    3. **Iterative Improvement**: Past lessons inform future predictions

    ### 🔄 The Process

    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Baseline ReAct Agent
        ```
        1. Receive task
        2. Reason with tools
        3. Make prediction
        4. Done (no learning)
        ```
        """)

    with col2:
        st.markdown("""
        #### Reflexion Agent
        ```
        Trial 1:
        1. Receive task
        2. Reason with tools
        3. Make prediction
        4. Evaluate result
        5. If failed: Generate reflection
        6. Store in memory

        Trial 2:
        1. Retrieve relevant past reflections
        2. Reason with tools + memory
        3. Make improved prediction
        4. Usually succeeds!
        ```
        """)

    st.markdown("---")

    st.markdown("""
    ### 🧠 Example Reflection

    **Task:** Predict sales for Store 5, Department 12, Week 48

    **Trial 1:**
    - Prediction: $25,000
    - Actual: $35,000
    - Error: 28.6%

    **Reflection Generated:**
    > "I used a simple 4-week average but failed to account for the holiday season in Week 48.
    > The seasonality check showed a 30% boost during November-December, which I overlooked.
    > For future holiday season predictions, I should weight recent holiday weeks more heavily."

    **Trial 2:**
    - Retrieved reflection about holiday seasonality
    - Applied seasonal adjustment
    - Prediction: $34,500
    - Actual: $35,000
    - Error: 1.4% ✅

    ### 📚 Research Background

    Reflexion is based on the paper:
    **"Reflexion: Language Agents with Verbal Reinforcement Learning"**
    (Shinn et al., NeurIPS 2023)

    Instead of traditional reinforcement learning with numeric rewards, Reflexion uses
    **verbal feedback** (natural language reflections) to guide learning.

    ### 🎓 Key Benefits

    1. **Interpretable**: Reflections are human-readable
    2. **Sample Efficient**: Learns from few examples
    3. **Transferable**: Insights apply to similar tasks
    4. **Debuggable**: Can see exactly what the agent learned

    ### 🛠️ Implementation Details

    **Tools Used:**
    - LangChain for agent orchestration
    - OpenAI GPT-3.5/4 for reasoning and reflection
    - Simple keyword-based memory retrieval
    - Streamlit for interactive demonstration

    **Dataset:**
    - Walmart sales data with weekly sales by store/department
    - Includes seasonal patterns and trends
    - 10-20 test cases for evaluation
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Reflexion Business Intelligence Agent | AI Lab Final Project</p>
    <p>Built with LangChain, OpenAI, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
