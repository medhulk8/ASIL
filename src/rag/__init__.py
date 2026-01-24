"""
RAG (Retrieval-Augmented Generation) module for ASIL project.

Provides web search capabilities to augment football predictions with
real-time information about injuries, transfers, team news, etc.
"""

from src.rag.web_search_rag import WebSearchRAG

__all__ = ["WebSearchRAG"]
