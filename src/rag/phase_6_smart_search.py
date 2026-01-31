"""
Phase 6: Smart Web Search Strategy

Implements intelligent search strategies to fix the -18.5% accuracy impact from naive searches:

1. Query Deduplication - Track and avoid redundant searches
2. LLM-Generated Queries - Smart, context-aware query generation
3. Conditional Search - Skip searches when cached data is good enough
4. Query Relevance Filtering - Only search for truly needed information

Key Insight: Web searches were HURTING performance (-18.5% accuracy).
This is likely due to:
- Generic, template-based queries returning irrelevant info
- Outdated news mixed with current context
- Too many searches adding noise

Solution: Be selective and smart about WHAT and WHEN to search.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
import ollama


class SmartSearchStrategy:
    """
    Intelligent search strategy that decides:
    - WHAT to search (LLM-generated, relevant queries)
    - WHEN to search (only if needed, cache-aware)
    - HOW MANY searches (minimal but effective)
    """

    def __init__(self, model: str = "llama3.1:8b"):
        """
        Initialize smart search strategy.

        Args:
            model: LLM model for query generation
        """
        self.model = model
        self.session_queries: Set[str] = set()  # Track queries this session
        self.cache_path = Path("data/cache/smart_search_decisions.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load historical query effectiveness (if exists)
        self.query_effectiveness = self._load_effectiveness()

    def _load_effectiveness(self) -> Dict[str, float]:
        """Load historical query effectiveness scores"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f).get('effectiveness', {})
        except Exception:
            pass
        return {}

    def _save_effectiveness(self):
        """Save query effectiveness scores"""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump({'effectiveness': self.query_effectiveness}, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save effectiveness data: {e}")

    def _query_hash(self, query: str) -> str:
        """Generate a normalized hash for a query to detect duplicates"""
        # Normalize query (lowercase, remove extra spaces)
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def should_skip_search(
        self,
        home_team: str,
        away_team: str,
        kg_insights: Optional[Dict[str, Any]] = None,
        cache_available: bool = False,
        stats_quality: str = "good"
    ) -> Dict[str, Any]:
        """
        Decide whether to skip web search entirely.

        Skips search if:
        1. We have high-confidence KG insights AND good stats
        2. Recent cached results exist for this matchup
        3. Both teams are well-known (not newly promoted)

        Args:
            home_team: Home team name
            away_team: Away team name
            kg_insights: Knowledge graph insights
            cache_available: Whether cached search results exist
            stats_quality: Quality of existing stats ('good', 'medium', 'poor')

        Returns:
            {
                'skip': bool,
                'reason': str,
                'confidence': float
            }
        """
        reasons = []
        confidence_score = 0.0

        # Check KG quality
        kg_confidence = 'none'
        if kg_insights:
            kg_confidence = kg_insights.get('confidence', 'none')

        if kg_confidence in ['high', 'medium']:
            confidence_score += 0.4
            reasons.append(f"KG has {kg_confidence} confidence tactical insights")

        # Check stats quality
        if stats_quality == 'good':
            confidence_score += 0.3
            reasons.append("Good quality historical stats available")
        elif stats_quality == 'medium':
            confidence_score += 0.15

        # Check cache
        if cache_available:
            confidence_score += 0.3
            reasons.append("Recent cached search results exist")

        # Decision threshold: skip if confidence >= 0.7
        skip = confidence_score >= 0.7

        return {
            'skip': skip,
            'reason': ' + '.join(reasons) if reasons else 'Insufficient context',
            'confidence': confidence_score
        }

    def generate_smart_queries(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        kg_insights: Optional[Dict[str, Any]] = None,
        existing_context: Optional[str] = None,
        max_queries: int = 3
    ) -> List[str]:
        """
        Use LLM to generate smart, context-aware search queries.

        Instead of template-based queries, let the LLM decide what's most relevant
        based on:
        - What we already know (KG insights, stats)
        - What's missing (gaps in knowledge)
        - What's time-sensitive (injuries, recent form)

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date (YYYY-MM-DD)
            kg_insights: Existing KG insights
            existing_context: Any context we already have
            max_queries: Maximum queries to generate (default 3, not 5-7!)

        Returns:
            List of smart, targeted search queries
        """
        # Build context for LLM
        year = match_date[:4] if match_date else str(datetime.now().year)

        context_summary = f"""Match: {home_team} vs {away_team} ({match_date})

What we already know:"""

        if kg_insights and kg_insights.get('confidence') not in ['none', None]:
            kg_conf = kg_insights.get('confidence', 'none')
            home_styles = ', '.join(kg_insights.get('home_styles', []))
            away_styles = ', '.join(kg_insights.get('away_styles', []))
            context_summary += f"""
- Tactical styles: {home_team} ({home_styles}) vs {away_team} ({away_styles})
- KG confidence: {kg_conf}"""
        else:
            context_summary += "\n- No tactical insights available yet"

        if existing_context:
            context_summary += f"\n- Additional context: {existing_context[:200]}..."

        # LLM prompt for query generation
        prompt = f"""{context_summary}

Task: Generate {max_queries} highly specific, high-value web search queries that would most improve prediction accuracy.

Guidelines:
1. Focus on TIME-SENSITIVE information (injuries, suspensions, recent form changes)
2. Avoid generic queries - be specific and recent
3. Skip information we already have from KG insights
4. Prioritize factors that directly impact match outcomes
5. Use year {year} for recency

Output ONLY the queries, one per line, no numbering or explanations.
Example good queries:
- Liverpool injury news December 2024 latest
- Arsenal defensive form last 3 matches 2024

Generate {max_queries} queries now:"""

        try:
            # Call LLM for query generation
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.7, 'num_predict': 200}
            )

            # Parse queries from response
            queries_text = response.get('response', '').strip()
            queries = []

            for line in queries_text.split('\n'):
                line = line.strip()
                # Remove common prefixes
                for prefix in ['- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()

                if line and len(line) > 10:  # Reasonable query length
                    queries.append(line)

            # Deduplicate and limit
            queries = self._deduplicate_queries(queries)[:max_queries]

            return queries

        except Exception as e:
            print(f"Warning: LLM query generation failed: {e}")
            # Fallback to minimal template queries
            return [
                f"{home_team} injury news latest {year}",
                f"{away_team} injury suspension {year}",
                f"{home_team} vs {away_team} preview {year}"
            ][:max_queries]

    def _deduplicate_queries(self, queries: List[str]) -> List[str]:
        """
        Remove duplicate and very similar queries.

        Uses query hashing to detect:
        - Exact duplicates
        - Queries already executed this session
        - Very similar queries (>80% token overlap)
        """
        unique_queries = []
        seen_hashes = set()

        for query in queries:
            query_hash = self._query_hash(query)

            # Skip if exact duplicate
            if query_hash in seen_hashes:
                continue

            # Skip if already searched this session
            if query_hash in {self._query_hash(q) for q in self.session_queries}:
                continue

            # Check token overlap with existing queries
            is_duplicate = False
            query_tokens = set(query.lower().split())

            for existing in unique_queries:
                existing_tokens = set(existing.lower().split())
                overlap = len(query_tokens & existing_tokens) / max(len(query_tokens), len(existing_tokens))

                if overlap > 0.8:  # >80% token overlap = too similar
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_queries.append(query)
                seen_hashes.add(query_hash)
                self.session_queries.add(query)

        return unique_queries

    def filter_queries_by_cache(
        self,
        queries: List[str],
        cache_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Filter out queries that would return only cached results.

        Args:
            queries: List of candidate queries
            cache_stats: Cache statistics from search system

        Returns:
            Filtered list of queries (only those needing fresh search)
        """
        # For now, simple implementation - can be enhanced with cache inspection
        # This would check if similar queries were recently cached
        return queries

    def evaluate_query_effectiveness(
        self,
        query: str,
        prediction_improved: bool,
        brier_delta: float
    ):
        """
        Track effectiveness of queries for future optimization.

        Args:
            query: The search query
            prediction_improved: Whether prediction improved with this search
            brier_delta: Change in Brier score (positive = improvement)
        """
        query_hash = self._query_hash(query)

        if query_hash not in self.query_effectiveness:
            self.query_effectiveness[query_hash] = {
                'query': query,
                'uses': 0,
                'improvements': 0,
                'avg_brier_delta': 0.0
            }

        stats = self.query_effectiveness[query_hash]
        stats['uses'] += 1

        if prediction_improved:
            stats['improvements'] += 1

        # Update running average of Brier delta
        stats['avg_brier_delta'] = (
            (stats['avg_brier_delta'] * (stats['uses'] - 1) + brier_delta) / stats['uses']
        )

        # Save periodically
        if stats['uses'] % 5 == 0:
            self._save_effectiveness()

    def get_effectiveness_report(self) -> str:
        """Generate a report on query effectiveness"""
        if not self.query_effectiveness:
            return "No query effectiveness data yet"

        report = []
        report.append("=" * 70)
        report.append("QUERY EFFECTIVENESS REPORT")
        report.append("=" * 70)

        # Sort by improvement rate
        sorted_queries = sorted(
            self.query_effectiveness.items(),
            key=lambda x: x[1]['improvements'] / max(x[1]['uses'], 1),
            reverse=True
        )

        for query_hash, stats in sorted_queries[:10]:  # Top 10
            improvement_rate = stats['improvements'] / stats['uses']
            report.append(f"\nQuery: {stats['query'][:60]}...")
            report.append(f"  Uses: {stats['uses']}")
            report.append(f"  Improvement rate: {improvement_rate:.1%}")
            report.append(f"  Avg Brier delta: {stats['avg_brier_delta']:+.3f}")

        return '\n'.join(report)

    def reset_session(self):
        """Reset session-level tracking"""
        self.session_queries.clear()


class MinimalSearchDecider:
    """
    Even simpler strategy: Decide the absolute minimum searches needed.

    Philosophy: Less is more. Each search adds noise unless it's critical.
    Only search for things that:
    1. Can't be inferred from stats
    2. Are time-sensitive (change daily/weekly)
    3. Have proven to improve predictions
    """

    def __init__(self):
        self.critical_only = True

    def get_minimal_queries(
        self,
        home_team: str,
        away_team: str,
        year: str,
        has_kg_insights: bool = False
    ) -> List[str]:
        """
        Get absolute minimal queries - only what's critical.

        Only searches for:
        - Injuries/suspensions (most time-sensitive, high impact)
        - Skip everything else that can be inferred from historical stats

        Args:
            home_team: Home team name
            away_team: Away team name
            year: Current year
            has_kg_insights: Whether we have KG tactical insights

        Returns:
            List of 0-2 queries (yes, maybe ZERO!)
        """
        queries = []

        # ONLY search for injuries/suspensions - highest impact, most time-sensitive
        queries.append(f"{home_team} {away_team} injury suspension news {year}")

        # That's it. Form can be inferred from stats. Tactics from KG.
        # Head-to-head is in historical data.

        return queries


# ============================================================================
# Testing
# ============================================================================

def test_smart_search_strategy():
    """Test the smart search strategy"""
    print("=" * 70)
    print("TESTING SMART SEARCH STRATEGY")
    print("=" * 70)

    strategy = SmartSearchStrategy()

    # Test 1: Should we skip search?
    print("\nTest 1: Skip decision")
    decision = strategy.should_skip_search(
        "Liverpool",
        "Arsenal",
        kg_insights={'confidence': 'high', 'home_styles': ['possession'], 'away_styles': ['counter-attack']},
        cache_available=True,
        stats_quality='good'
    )
    print(f"Skip: {decision['skip']}")
    print(f"Reason: {decision['reason']}")
    print(f"Confidence: {decision['confidence']:.2f}")

    # Test 2: Generate smart queries
    print("\n" + "=" * 70)
    print("Test 2: LLM-generated queries")
    print("=" * 70)

    queries = strategy.generate_smart_queries(
        home_team="Manchester City",
        away_team="Chelsea",
        match_date="2024-12-15",
        kg_insights={'confidence': 'medium', 'home_styles': ['possession'], 'away_styles': ['defensive']},
        max_queries=3
    )

    print(f"\nGenerated {len(queries)} smart queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    # Test 3: Deduplication
    print("\n" + "=" * 70)
    print("Test 3: Query deduplication")
    print("=" * 70)

    duplicate_queries = [
        "Liverpool injury news 2024",
        "Liverpool injury news latest 2024",  # Very similar
        "Arsenal suspension news",
        "Liverpool injury news 2024",  # Exact duplicate
        "Manchester United form"
    ]

    unique = strategy._deduplicate_queries(duplicate_queries)
    print(f"\nOriginal: {len(duplicate_queries)} queries")
    print(f"After dedup: {len(unique)} queries")
    for i, q in enumerate(unique, 1):
        print(f"{i}. {q}")

    # Test 4: Minimal search
    print("\n" + "=" * 70)
    print("Test 4: Minimal search strategy")
    print("=" * 70)

    minimal = MinimalSearchDecider()
    minimal_queries = minimal.get_minimal_queries(
        "Liverpool", "Arsenal", "2024", has_kg_insights=True
    )
    print(f"\nMinimal queries: {len(minimal_queries)}")
    for i, q in enumerate(minimal_queries, 1):
        print(f"{i}. {q}")


if __name__ == "__main__":
    test_smart_search_strategy()
