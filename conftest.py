"""
Root conftest.py — adds src/ to sys.path so tests can import llada_fast
without requiring a `pip install -e .` step.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
