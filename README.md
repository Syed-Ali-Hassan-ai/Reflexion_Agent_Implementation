# 🧠 Reflexion Business Intelligence Agent

A demonstration system implementing **Reflexion** (verbal reinforcement learning) for business intelligence tasks, specifically sales forecasting. This project showcases how AI agents can learn from their mistakes through self-reflection and episodic memory.

## 🎯 Project Overview

Traditional ReAct agents make predictions but don't learn from failures. **Reflexion** adds:
1. **Self-Reflection**: Agent analyzes WHY it failed
2. **Episodic Memory**: Stores reflections for future use
3. **Iterative Improvement**: Gets better over multiple trials

This implementation applies Reflexion to **sales forecasting** using Walmart sales data, demonstrating measurable improvement through self-reflection.

## 🌟 Key Features

- **Baseline ReAct Agent**: Standard reasoning and action agent for sales forecasting
- **Reflexion Agent**: Enhanced agent with self-reflection and memory
- **Interactive Streamlit Interface**: Professional demo with multiple views
- **Batch Comparison**: Side-by-side performance evaluation
- **Memory Explorer**: Visualize stored reflections and insights
- **Real-time Metrics**: Track improvement across trials

## 📁 Project Structure

```
reflexion-business-agent/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── agents/
│   ├── __init__.py
│   ├── baseline_agent.py          # Standard ReAct agent
│   ├── reflexion_agent.py         # Reflexion-enhanced agent
│   └── tools.py                   # Sales analysis tools
├── memory/
│   ├── __init__.py
│   └── episodic_memory.py         # Memory storage/retrieval
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Walmart data loader
│   ├── evaluator.py               # Prediction evaluator
│   └── prompts.py                 # LLM prompt templates
└── data/
    └── (Walmart dataset - auto-downloaded)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Reflexion_Agent_Implementation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key:**

You can either:
- Enter it in the Streamlit sidebar when running the app (recommended for demo)
- Create a `.env` file in the project root:
  ```
  OPENAI_API_KEY=your-api-key-here
  ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Live Demo Tab

**Generate and Test Individual Predictions:**

1. Enter your OpenAI API key in the sidebar
2. Click "Generate Random Test Case" to get a sales forecasting scenario
3. Review the historical sales data and trends
4. Select agent type (Baseline or Reflexion)
5. Click "Run Agent" to see the prediction process
6. View the results, reasoning trace, and (for Reflexion) self-reflections

**Understanding Results:**
- ✅ Success: Prediction within ±10% of actual sales
- ❌ Failure: Prediction exceeds ±10% threshold
- Reflexion agents improve from Trial 1 to Trial 2 using stored reflections

### 2. Batch Comparison Tab

**Compare Agent Performance:**

1. Click "Run Batch Comparison"
2. Wait as both agents process 10 test cases
3. Review the detailed results table
4. Analyze summary statistics and visualizations
5. See error reduction and success rate improvements

**What to Look For:**
- Average error reduction (typically 10-20%)
- Success rate improvement
- Per-test-case error comparison
- Consistent Reflexion outperformance

### 3. Memory Explorer Tab

**Explore Stored Reflections:**

1. Run some predictions with the Reflexion agent first
2. View memory summary statistics
3. Expand individual memories to see:
   - Task description
   - Prediction vs actual
   - Reflection/insight learned
   - Success/failure status
4. Use search to find reflections by keyword

**Key Insights:**
- Failed predictions generate actionable reflections
- Successful predictions store positive examples
- Reflections are human-readable and interpretable

### 4. How It Works Tab

**Learn About Reflexion:**

- Understand the problem Reflexion solves
- See the process comparison (Baseline vs Reflexion)
- Review example reflections
- Read about the research background
- Understand implementation details

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model settings
DEFAULT_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
TEMPERATURE = 0.7

# Evaluation
SUCCESS_THRESHOLD = 10.0  # MAPE threshold (%)

# Memory
MAX_MEMORY_ITEMS = 100
MEMORY_RETRIEVAL_TOP_K = 3

# Agent
MAX_ITERATIONS = 5
AGENT_VERBOSE = True
```

## 📊 Dataset

The application uses the **Walmart sales dataset** from Kaggle, which contains:
- Weekly sales data for multiple stores and departments
- Historical trends and seasonal patterns
- Holiday indicators

**Auto-Loading:**
- The app automatically downloads the dataset using `kagglehub`
- If download fails, it uses generated sample data
- Sample data includes realistic trends and seasonality

## 🧪 Technical Details

### Architecture

**Baseline ReAct Agent:**
```
Task → Reason → Tools → Action → Observation → Prediction
```

**Reflexion Agent:**
```
Trial 1:
  Task → Reason → Tools → Action → Observation → Prediction
    ↓
  Evaluate → Failed? → Generate Reflection → Store in Memory

Trial 2:
  Retrieve Memories → Task + Context → Reason → Tools → Prediction
    ↓
  Evaluate → Improved!
```

### Tools Available to Agents

1. **calculate_average_sales**: Compute average sales over N weeks
2. **check_seasonality**: Identify seasonal patterns and trends
3. **get_historical_data**: Retrieve week-by-week sales data
4. **analyze_trends**: Analyze growth/decline trends
5. **make_prediction**: Make final prediction with reasoning

### Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error
- **Success**: MAPE < 10%
- **Improvement**: MAPE reduction between trials

## 📈 Expected Results

**Baseline Agent:**
- Average error: ~15-20%
- Success rate: ~40-50%
- No improvement between attempts

**Reflexion Agent:**
- Trial 1 error: ~15-20% (similar to baseline)
- Trial 2 error: ~8-12% (after reflection)
- Success rate: ~60-80%
- **10-30% average improvement** from Trial 1 to Trial 2

## 🎓 Research Background

This implementation is based on:

**"Reflexion: Language Agents with Verbal Reinforcement Learning"**
*Shinn et al., NeurIPS 2023*

Key innovation: Using **natural language reflections** instead of numeric rewards for reinforcement learning.

**Benefits:**
- Interpretable learning process
- Sample efficient (learns from few examples)
- Transferable knowledge across tasks
- Human-debuggable

## 🛠️ Technology Stack

- **Framework**: LangChain for agent orchestration
- **LLM**: OpenAI GPT-3.5-turbo or GPT-4
- **Frontend**: Streamlit for interactive demo
- **Memory**: Keyword-based episodic memory
- **Data**: Pandas, NumPy for data processing
- **Visualization**: Plotly for charts

## 💡 Use Cases

This approach can be extended to:

1. **Financial forecasting**: Stock prices, revenue predictions
2. **Demand planning**: Inventory optimization, supply chain
3. **Risk assessment**: Credit scoring, fraud detection
4. **Marketing analytics**: Campaign performance, customer behavior
5. **Any domain where learning from mistakes improves performance**

## 🐛 Troubleshooting

**Issue: API Key Error**
- Ensure your OpenAI API key is valid
- Check you have sufficient credits
- Try using GPT-3.5-turbo if GPT-4 fails

**Issue: Dataset Download Fails**
- The app will automatically use sample data
- No action needed - sample data is realistic

**Issue: Agent Takes Long Time**
- Reduce MAX_ITERATIONS in config.py
- Use GPT-3.5-turbo instead of GPT-4
- Expected: 30-60 seconds per prediction

**Issue: Poor Performance**
- Ensure TEMPERATURE is set correctly (0.7)
- Check that historical data has enough weeks
- Try different test cases

## 📝 Example Session Flow

1. **Instructor opens app** → Sees clean interface
2. **Enters API key** → System ready
3. **Generates test case** → "Predict Store 5, Dept 12, Week 48"
4. **Runs Baseline** → Prediction: $25,000 | Actual: $35,000 | Error: 28% ❌
5. **Switches to Reflexion** → Trial 1 similar failure
6. **Sees Reflection** → "Missed holiday seasonality boost"
7. **Trial 2 runs** → Uses reflection → Prediction: $34,500 | Error: 1.4% ✅
8. **Batch Comparison** → Reflexion outperforms on 8/10 cases
9. **Memory Explorer** → Reviews learned insights
10. **Result**: Impressed by self-improvement capability! 🎉

## 🎯 Success Criteria

- ✅ Clear demonstration of self-reflection
- ✅ Measurable improvement between trials
- ✅ Interpretable reflections
- ✅ Professional interface
- ✅ Easy to understand for non-experts
- ✅ Reproducible results

## 📜 License

This is an educational project for AI Lab demonstration purposes.

## 🤝 Contributing

This is a course project, but suggestions are welcome:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions about this project, please contact [your email/info]

---

## 🎓 Academic Context

**Course**: AI Lab Final Project
**Topic**: Reflexion - Verbal Reinforcement Learning
**Goal**: Demonstrate self-improving agents for business intelligence

**Key Learning Objectives:**
1. Understand limitations of standard ReAct agents
2. Implement self-reflection mechanisms
3. Design episodic memory systems
4. Evaluate agent improvement quantitatively
5. Create professional demonstrations

---

**Built with ❤️ using LangChain, OpenAI, and Streamlit**
