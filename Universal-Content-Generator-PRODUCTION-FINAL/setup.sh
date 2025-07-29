#!/bin/bash

# Universal Content Generator - Setup Script
# This script installs all required dependencies and downloads the spaCy model

echo "🚀 Universal Content Generator - Setup Script"
echo "============================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

# Display Python version
echo "✅ Python version:"
python3 --version
echo ""

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Linux/Mac
        source venv/bin/activate
    fi
    echo "✅ Virtual environment created and activated"
    echo ""
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# Install requirements
echo "📦 Installing requirements..."
pip3 install -r requirements.txt
echo ""

# Download spaCy model
echo "📦 Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it and add your OpenAI API key."
    else
        echo "❌ .env.example not found. Please create a .env file with your OPENAI_API_KEY."
    fi
else
    echo "✅ .env file already exists"
fi
echo ""

# Success message
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run the app with: streamlit run app.py"
echo ""
echo "For deployment:"
echo "- Render: Push to GitHub and connect via Render dashboard"
echo "- Heroku: Use 'heroku create' and 'git push heroku main'"
echo "- Docker: Use 'docker build -t universal-content-generator .'"
echo ""