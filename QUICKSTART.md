# 🚀 Quick Start Guide

Get your Reflexion Business Intelligence Agent running in 3 minutes!

## Step 1: Install Dependencies

**Option A: Using setup script (Linux/Mac)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Manual installation**
```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key (you'll need it in the next step)

## Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 4: Use the Demo

1. **Enter your API key** in the sidebar
2. **Go to "Live Demo" tab**
3. Click **"Generate Random Test Case"**
4. Select **"Reflexion Agent"** in the sidebar
5. Click **"Run Agent"**
6. Watch the magic happen! ✨

## What You'll See

- **Trial 1**: Agent makes a prediction (might fail)
- **Reflection**: Agent analyzes what went wrong
- **Trial 2**: Agent tries again with learned insights (usually succeeds!)

## Example Flow

```
Test Case: Predict sales for Store 5, Dept 12, Week 48
Actual Sales: $35,000

Trial 1:
  Prediction: $25,000
  Error: 28.6% ❌
  Reflection: "I missed the holiday season boost..."

Trial 2:
  Prediction: $34,500 (using past reflection)
  Error: 1.4% ✅
  SUCCESS!
```

## Troubleshooting

**"No module named..."**
- Run: `pip install -r requirements.txt`

**"Invalid API key"**
- Check your OpenAI API key is correct
- Ensure you have credits in your OpenAI account

**"Agent takes too long"**
- Normal! Each prediction takes 30-60 seconds
- GPT-3.5-turbo is faster than GPT-4

## Next Steps

- Try the **Batch Comparison** tab to see overall performance
- Explore **Memory Explorer** to see what the agent learned
- Read **How It Works** to understand Reflexion

## Need Help?

- Check the full [README.md](README.md)
- Review code comments
- Check OpenAI API status

---

**Enjoy building self-improving AI agents! 🧠**
