#!/bin/bash
# Setup script for crypto pattern recognition project

echo "🔧 Setting up crypto pattern recognition environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️ No virtual environment detected."
    echo "To create a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    echo "If python3-venv is not available, install it:"
    echo "  sudo apt install python3-venv python3-pip"
    echo ""
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
if pip install -r requirements.txt; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
fi

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data
echo "✅ Data directory created"

# Make scripts executable
echo "🔧 Setting script permissions..."
chmod +x collect_data.py test_collection.py
echo "✅ Script permissions set"

echo ""
echo "🎉 Setup complete!"
echo ""