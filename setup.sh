#!/bin/bash

# Setup script for Reflexion Business Intelligence Agent

echo "======================================"
echo "Reflexion Agent Setup"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv

    echo "Activating virtual environment..."
    source venv/bin/activate

    echo "Virtual environment created and activated."
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Setup Complete!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "1. Get your OpenAI API key from: https://platform.openai.com/api-keys"
    echo "2. Run the application: streamlit run app.py"
    echo "3. Enter your API key in the sidebar"
    echo ""
    echo "Enjoy your Reflexion Agent! 🧠"
else
    echo ""
    echo "Error: Installation failed. Please check the error messages above."
    exit 1
fi
