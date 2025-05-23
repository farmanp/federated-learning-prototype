#!/bin/bash
# Script to set up the virtual environment with Python 3.9

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 is not installed. Installing..."
    # For macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS. Installing Python 3.9 using Homebrew..."
        brew install python@3.9
    else
        echo "Please install Python 3.9 manually."
        exit 1
    fi
fi

# Create a virtual environment
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"

echo "Virtual environment is set up successfully with Python 3.9!"
echo "To activate the virtual environment, run: source venv/bin/activate"
