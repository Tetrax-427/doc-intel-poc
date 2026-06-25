"""
classification/ — E1 two-stage document classifier.

Public API (used by retrieval.py):
    from classification.pipeline import classify
"""
from classification.pipeline import classify

__all__ = ["classify"]