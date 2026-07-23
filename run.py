#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple launcher for the optional Gradio document-Q&A interface.
The primary Contract Risk Reviewer front end is streamlit_app.py.
"""

import sys
import os
import io

# Fix Windows encoding issues - force UTF-8 for console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ui import launch
    
    if __name__ == "__main__":
        print("Starting document Q&A interface...")
        launch()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
