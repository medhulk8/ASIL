"""
Web Search RAG for Football Predictions

This module provides web search capabilities using the Tavily API
to retrieve real-time information about teams, players, injuries,
and other relevant football context.

Features:
- Automatic query generation for match context
- Search result caching to reduce API calls
- Relevance-based result filtering
- Formatted context for LLM consumption

Usage:
    from src.rag import WebSearchRAG

    rag = WebSearchRAG(tavily_api_key="your_api_key")
    context = rag.get_match_context("Liverpool", "Arsenal", "2024-01-15")
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class WebSearchRAG:
    """
    Web Search RAG client using Tavily API.

    Tavily provides high-quality search results optimized for AI/LLM use cases,
    with pre-extracted content snippets and relevance scoring.

    Attributes:
        tavily: TavilyClient instance
        search_cache: Dict caching search results by query
    """

    def __init__(self, tavily_api_key: Optional[str] = None):
        """
        Initialize WebSearchRAG with Tavily API.

        Args:
            tavily_api_key: API key from tavily.com
                           If not provided, reads from TAVILY_API_KEY env var
                           Free tier: 1000 searches/month

        Raises:
            ValueError: If no API key provided or found in environment
        """
        from tavily import TavilyClient

        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "Tavily API key required. Either pass tavily_api_key parameter "
                "or set TAVILY_API_KEY environment variable. "
                "Get a free key at: https://tavily.com"
            )

        self.tavily = TavilyClient(api_key=api_key)
        self.search_cache: Dict[str, Dict[str, Any]] = {}

    def execute_searches(
        self,
        queries: List[str],
        max_searches: int = 5,
        max_results_per_query: int = 5
    ) -> Dict[str, Any]:
        """
        Execute multiple searches using Tavily API.

        For each query:
        1. Check cache first (avoid duplicate API calls)
        2. If not cached, call Tavily search API
        3. Extract relevant content from response
        4. Cache result for future use

        Args:
            queries: List of search queries to execute
            max_searches: Maximum number of searches to perform
            max_results_per_query: Maximum results per search (1-10)

        Returns:
            Dictionary containing:
            - queries_executed: Number of searches performed
            - cached_hits: Number of cache hits
            - results: Dict mapping query -> list of results
            - all_content: Combined content for LLM context

        Example:
            >>> rag = WebSearchRAG(api_key="...")
            >>> results = rag.execute_searches(["Liverpool injuries 2024"])
            >>> print(results["all_content"])
        """
        results = {}
        cached_hits = 0
        queries_to_execute = queries[:max_searches]

        for query in queries_to_execute:
            # Check cache first
            if query in self.search_cache:
                results[query] = self.search_cache[query]
                cached_hits += 1
                continue

            try:
                # Execute Tavily search
                response = self.tavily.search(
                    query=query,
                    max_results=max_results_per_query,
                    search_depth="basic"  # Use "advanced" for deeper search
                )

                # Extract and format results
                formatted_results = []
                for item in response.get("results", []):
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "score": item.get("score", 0.0)
                    })

                # Sort by relevance score
                formatted_results.sort(key=lambda x: x["score"], reverse=True)

                # Cache and store
                self.search_cache[query] = formatted_results
                results[query] = formatted_results

            except Exception as e:
                results[query] = {"error": str(e)}

        # Combine all content for LLM context
        all_content = self._format_for_llm(results)

        return {
            "queries_executed": len(queries_to_execute) - cached_hits,
            "cached_hits": cached_hits,
            "results": results,
            "all_content": all_content
        }

    def _format_for_llm(self, results: Dict[str, Any]) -> str:
        """
        Format search results into a clean context string for LLM.

        Args:
            results: Dictionary of query -> results

        Returns:
            Formatted string suitable for including in LLM prompts
        """
        sections = []

        for query, items in results.items():
            if isinstance(items, dict) and "error" in items:
                continue

            if not items:
                continue

            section = f"### Search: {query}\n"
            for i, item in enumerate(items[:3], 1):  # Top 3 per query
                title = item.get("title", "Untitled")
                content = item.get("content", "")
                url = item.get("url", "")

                # Truncate content if too long
                if len(content) > 500:
                    content = content[:500] + "..."

                section += f"\n**{i}. {title}**\n"
                section += f"{content}\n"
                section += f"Source: {url}\n"

            sections.append(section)

        if not sections:
            return "No relevant web search results found."

        return "\n---\n".join(sections)

    def generate_match_queries(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None
    ) -> List[str]:
        """
        Generate relevant search queries for a match.

        Creates queries covering:
        - Team recent form and results
        - Injury news
        - Team news and lineup
        - Head-to-head history
        - Tactical analysis

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Optional match date (YYYY-MM-DD format)

        Returns:
            List of search queries (typically 4-6 queries)
        """
        year = datetime.now().year if not match_date else match_date[:4]

        queries = [
            f"{home_team} recent form results {year}",
            f"{away_team} recent form results {year}",
            f"{home_team} injuries team news",
            f"{away_team} injuries team news",
            f"{home_team} vs {away_team} preview prediction",
        ]

        return queries

    def get_match_context(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        max_searches: int = 5
    ) -> Dict[str, Any]:
        """
        Get comprehensive web context for a match.

        Convenience method that:
        1. Generates relevant queries for the match
        2. Executes searches
        3. Returns formatted context

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Optional match date (YYYY-MM-DD)
            max_searches: Maximum searches to perform

        Returns:
            Dictionary with search results and formatted context

        Example:
            >>> rag = WebSearchRAG(api_key="...")
            >>> context = rag.get_match_context("Liverpool", "Arsenal")
            >>> print(context["all_content"])
        """
        queries = self.generate_match_queries(home_team, away_team, match_date)
        return self.execute_searches(queries, max_searches=max_searches)

    def search_single(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a single search query.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of search result dictionaries
        """
        result = self.execute_searches([query], max_results_per_query=max_results)
        return result["results"].get(query, [])

    def clear_cache(self):
        """Clear the search result cache."""
        self.search_cache.clear()


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print(" WEB SEARCH RAG TEST (Tavily)")
    print("=" * 70)

    # Get API key from environment or command line
    api_key = os.environ.get("TAVILY_API_KEY")

    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key:
        print("\nNo API key found!")
        print("\nTo run tests, either:")
        print("  1. Set TAVILY_API_KEY environment variable")
        print("  2. Pass API key as command line argument:")
        print("     python web_search_rag.py YOUR_API_KEY")
        print("\nGet a free API key at: https://tavily.com")
        sys.exit(1)

    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")

    # Initialize RAG
    print("\nInitializing WebSearchRAG...")
    rag = WebSearchRAG(tavily_api_key=api_key)
    print("Initialized successfully!")

    # Test 1: Single search
    print("\n" + "=" * 70)
    print(" TEST 1: Single Search - Liverpool Recent Form")
    print("=" * 70)

    query = "Liverpool recent form 2024 Premier League"
    print(f"\nQuery: {query}")
    print("\nSearching...")

    results = rag.search_single(query, max_results=5)

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   URL: {result['url']}")
        content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"   Content: {content_preview}")

    # Test 2: Match context
    print("\n" + "=" * 70)
    print(" TEST 2: Full Match Context - Liverpool vs Arsenal")
    print("=" * 70)

    print("\nGenerating match context...")
    context = rag.get_match_context("Liverpool", "Arsenal", max_searches=3)

    print(f"\nQueries executed: {context['queries_executed']}")
    print(f"Cache hits: {context['cached_hits']}")

    print("\n" + "-" * 70)
    print(" FORMATTED CONTEXT FOR LLM")
    print("-" * 70)
    print(context["all_content"])

    # Test 3: Cache test
    print("\n" + "=" * 70)
    print(" TEST 3: Cache Test")
    print("=" * 70)

    print("\nRunning same query again...")
    context2 = rag.get_match_context("Liverpool", "Arsenal", max_searches=3)
    print(f"Queries executed: {context2['queries_executed']} (should be 0)")
    print(f"Cache hits: {context2['cached_hits']} (should be 3)")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED!")
    print("=" * 70)
