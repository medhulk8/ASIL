"""
Knowledge Graph Module for Football Tactical Analysis

Provides two knowledge graph implementations:
- FootballKnowledgeGraph: Static, hardcoded team-style mappings
- DynamicKnowledgeGraph: LLM-powered, evidence-based style classification
"""

from .football_knowledge_graph import FootballKnowledgeGraph
from .dynamic_knowledge_graph import DynamicKnowledgeGraph

__all__ = ['FootballKnowledgeGraph', 'DynamicKnowledgeGraph']
