#!/bin/bash
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""
echo "Available commands:"
echo "  python train.py --data-dir data/training --epochs 50"
echo "  python retrain_model.py"
echo "  streamlit run streamlit_app.py"
echo ""
