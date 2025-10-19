#!/bin/bash

echo "🌊 Underwater Acoustic Classification System"
echo "=============================================="
echo ""

if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

echo "🚀 Starting Streamlit UI..."
echo ""
echo "📱 The application will open in your browser"
echo "🔗 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app.py
