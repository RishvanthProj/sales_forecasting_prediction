"""
Data Cleaner Compatibility Module.
Re-exports clean_dataset from src.cleaner for backward compatibility.
"""

from src.cleaner import clean_dataset

__all__ = ["clean_dataset"]
