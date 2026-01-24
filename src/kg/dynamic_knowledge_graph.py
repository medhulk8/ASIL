"""
Dynamic Knowledge Graph with LLM-Powered Team Style Classification

This module provides a knowledge graph that dynamically generates team tactical
profiles using web search + LLM classification, rather than hardcoded mappings.

Key Features:
- LLM-based team style classification (Ollama)
- Web search for current tactical information (Tavily)
- SQLite caching to avoid repeated API calls
- Universal tactical counter relationships
- Evidence-based style assignments

Usage:
    from src.kg.dynamic_knowledge_graph import DynamicKnowledgeGraph
    from src.rag.web_search_rag import WebSearchRAG

    web_rag = WebSearchRAG(tavily_api_key="...")
    kg = DynamicKnowledgeGraph("data/processed/asil.db", web_rag)

    styles = kg.get_team_style("Liverpool")
    matchup = kg.get_tactical_matchup("Liverpool", "Arsenal")
"""

import json
import os
import re
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"


class DynamicKnowledgeGraph:
    """
    Dynamic knowledge graph with LLM-powered team style classification.

    Unlike static knowledge graphs with hardcoded team-style mappings, this
    implementation dynamically generates styles using:
    1. Database statistics (shots, goals, corners)
    2. Web search for current tactical analysis
    3. LLM classification based on evidence

    Results are cached in SQLite to avoid repeated API calls.

    Attributes:
        db_path: Path to SQLite database
        web_rag: WebSearchRAG instance for web searches
        model: Ollama model name
        ollama: Ollama client
        TACTICAL_COUNTERS: Universal tactical relationships
    """

    # Universal tactical counter relationships (football theory)
    TACTICAL_COUNTERS = {
        "HIGH_PRESS": {
            "countered_by": ["LOW_BLOCK", "LONG_BALL"],
            "effective_against": ["POSSESSION", "TECHNICAL"]
        },
        "LOW_BLOCK": {
            "countered_by": ["WING_PLAY", "SET_PIECES"],
            "effective_against": ["HIGH_PRESS", "POSSESSION"]
        },
        "POSSESSION": {
            "countered_by": ["COUNTER_ATTACK", "HIGH_PRESS"],
            "effective_against": ["PHYSICAL", "LOW_BLOCK"]
        },
        "COUNTER_ATTACK": {
            "countered_by": ["LOW_BLOCK"],
            "effective_against": ["HIGH_PRESS", "POSSESSION"]
        },
        "WING_PLAY": {
            "countered_by": ["PHYSICAL"],
            "effective_against": ["LOW_BLOCK", "COUNTER_ATTACK"]
        },
        "SET_PIECES": {
            "countered_by": ["PHYSICAL"],
            "effective_against": ["LOW_BLOCK", "TECHNICAL"]
        },
        "PHYSICAL": {
            "countered_by": ["TECHNICAL", "POSSESSION"],
            "effective_against": ["TECHNICAL", "WING_PLAY"]
        },
        "TECHNICAL": {
            "countered_by": ["PHYSICAL", "HIGH_PRESS"],
            "effective_against": ["PHYSICAL"]
        },
        "LONG_BALL": {
            "countered_by": ["TECHNICAL"],
            "effective_against": ["HIGH_PRESS"]
        }
    }

    # Valid style names
    VALID_STYLES = set(TACTICAL_COUNTERS.keys())

    def __init__(
        self,
        db_path: str,
        web_rag,
        ollama_model: str = "llama3.2:3b",
        cache_days: int = 30
    ):
        """
        Initialize dynamic knowledge graph.

        Args:
            db_path: Path to SQLite database
            web_rag: WebSearchRAG instance for web searches
            ollama_model: Ollama model name for classification
            cache_days: How long to cache team styles before refresh
        """
        self.db_path = str(db_path)
        self.web_rag = web_rag
        self.model = ollama_model
        self.cache_days = cache_days

        # Initialize Ollama
        try:
            import ollama
            self.ollama = ollama
            # Test connection
            self.ollama.list()
            logger.info(f"DynamicKG: Ollama connected, model={ollama_model}")
        except Exception as e:
            logger.warning(f"DynamicKG: Ollama not available: {e}")
            self.ollama = None

        # Initialize cache table
        self._init_cache_table()

    def _init_cache_table(self):
        """Create table for caching LLM-generated team styles."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_team_styles (
                    team_name TEXT PRIMARY KEY,
                    styles TEXT NOT NULL,
                    confidence TEXT,
                    last_updated TEXT NOT NULL,
                    evidence TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.debug("DynamicKG: Cache table initialized")
        except Exception as e:
            logger.error(f"DynamicKG: Failed to create cache table: {e}")

    # =========================================================================
    # Core Methods
    # =========================================================================

    def get_team_style(
        self,
        team_name: str,
        force_refresh: bool = False
    ) -> List[str]:
        """
        Get team's playing style(s).

        Process:
        1. Check cache (unless force_refresh=True)
        2. If cached and fresh (<cache_days old), return cached
        3. Otherwise, generate new entry using LLM + web search
        4. Cache and return

        Args:
            team_name: Name of the team
            force_refresh: If True, ignore cache and regenerate

        Returns:
            List of style strings (1-2 styles per team)
        """
        if not force_refresh:
            cached = self._get_from_cache(team_name)
            if cached:
                return cached

        # Generate new entry
        return self._generate_and_cache_style(team_name)

    def _get_from_cache(self, team_name: str) -> Optional[List[str]]:
        """Check cache for team style."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT styles, last_updated FROM kg_team_styles WHERE team_name = ?",
                (team_name,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                styles_json, last_updated = row
                last_updated_dt = datetime.fromisoformat(last_updated)

                # Check if fresh
                if datetime.now() - last_updated_dt < timedelta(days=self.cache_days):
                    return json.loads(styles_json)

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def _generate_and_cache_style(self, team_name: str) -> List[str]:
        """Generate team style using LLM + web search, then cache."""
        print(f"  🔍 Generating tactical profile for {team_name}...")

        # Step 1: Get team statistics from database
        stats_context = self._get_team_stats_summary(team_name)

        # Step 2: Web search for tactical information
        web_context = ""
        if self.web_rag:
            try:
                search_results = self.web_rag.execute_searches(
                    [
                        f"{team_name} playing style tactics 2024",
                        f"{team_name} tactical analysis formation"
                    ],
                    max_searches=2,
                    max_results_per_query=3
                )
                web_context = search_results.get('all_content', '')
            except Exception as e:
                logger.warning(f"Web search failed for {team_name}: {e}")
                web_context = "Web search unavailable."

        # Step 3: LLM classification
        styles, confidence, evidence = self._classify_team_style(
            team_name,
            stats_context,
            web_context
        )

        # Step 4: Cache result
        self._save_to_cache(team_name, styles, confidence, evidence)

        print(f"  ✓ {team_name}: {', '.join(styles)} (confidence: {confidence})")

        return styles

    def _get_team_stats_summary(self, team_name: str) -> str:
        """Get statistical summary for team from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as matches,
                    AVG(CASE WHEN home_team = ? THEN home_shots ELSE away_shots END) as avg_shots,
                    AVG(CASE WHEN home_team = ? THEN home_goals ELSE away_goals END) as avg_goals,
                    AVG(CASE WHEN home_team = ? THEN home_corners ELSE away_corners END) as avg_corners,
                    SUM(CASE
                        WHEN (home_team = ? AND result = 'H') OR (away_team = ? AND result = 'A')
                        THEN 1 ELSE 0 END) as wins
                FROM matches
                WHERE (home_team = ? OR away_team = ?)
                ORDER BY date DESC
                LIMIT 20
            """, (team_name, team_name, team_name, team_name, team_name, team_name, team_name))

            row = cursor.fetchone()
            conn.close()

            if row and row[0] > 0:
                matches, avg_shots, avg_goals, avg_corners, wins = row
                win_rate = (wins / matches * 100) if matches > 0 else 0
                shots_str = f"{avg_shots:.1f}" if avg_shots else "N/A"
                goals_str = f"{avg_goals:.1f}" if avg_goals else "N/A"
                corners_str = f"{avg_corners:.1f}" if avg_corners else "N/A"
                return f"""Database Statistics (last {matches} matches):
- Shots per game: {shots_str}
- Goals per game: {goals_str}
- Corners per game: {corners_str}
- Win rate: {win_rate:.1f}%"""
            else:
                return "No statistical data available in database."

        except Exception as e:
            logger.warning(f"Stats query error: {e}")
            return "Statistics unavailable."

    def _classify_team_style(
        self,
        team_name: str,
        stats: str,
        web_context: str
    ) -> Tuple[List[str], str, str]:
        """
        Use LLM to classify team's playing style.

        Returns:
            Tuple of (styles, confidence, evidence)
        """
        if not self.ollama:
            # Fallback without LLM
            return self._fallback_classification(team_name, stats)

        # Truncate web context if too long
        if len(web_context) > 1500:
            web_context = web_context[:1500] + "\n[...truncated]"

        prompt = f"""Analyze {team_name}'s playing style based on the evidence below.

STATISTICAL CONTEXT:
{stats}

TACTICAL ANALYSIS FROM WEB:
{web_context}

AVAILABLE STYLES (choose 1-2 that BEST describe {team_name}):
- HIGH_PRESS: Aggressive pressing, wins ball high up pitch
- LOW_BLOCK: Defensive, compact shape, sits deep
- POSSESSION: Dominates ball, patient buildup
- COUNTER_ATTACK: Defends deep, fast transitions
- WING_PLAY: Attacks through wingers/full-backs
- SET_PIECES: Relies on set pieces for goals
- PHYSICAL: Direct, strong, aerial-focused
- TECHNICAL: Skillful, intricate passing

Respond in EXACTLY this format:
STYLES: STYLE1, STYLE2
CONFIDENCE: HIGH/MEDIUM/LOW
EVIDENCE: One sentence explaining why.

Example response:
STYLES: HIGH_PRESS, WING_PLAY
CONFIDENCE: HIGH
EVIDENCE: Liverpool consistently presses high and attacks through wide areas with Salah and Diaz."""

        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 200}
            )

            response_text = response['message']['content']
            return self._parse_classification_response(response_text, team_name)

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return self._fallback_classification(team_name, stats)

    def _parse_classification_response(
        self,
        response_text: str,
        team_name: str
    ) -> Tuple[List[str], str, str]:
        """Parse LLM classification response."""
        # Extract styles
        styles_match = re.search(
            r'STYLES?:\s*([A-Z_,\s]+?)(?:\n|CONFIDENCE)',
            response_text,
            re.IGNORECASE
        )

        if styles_match:
            styles_str = styles_match.group(1).strip()
            styles = [
                s.strip().upper().replace(' ', '_')
                for s in re.split(r'[,\s]+', styles_str)
                if s.strip()
            ]
            styles = [s for s in styles if s in self.VALID_STYLES][:2]
        else:
            # Fallback: find any valid style in response
            styles = [s for s in self.VALID_STYLES if s in response_text.upper()][:2]

        if not styles:
            styles = ["POSSESSION"]  # Default

        # Extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)',
            response_text,
            re.IGNORECASE
        )
        confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"

        # Extract evidence
        evidence_match = re.search(
            r'EVIDENCE:\s*(.+?)(?:\n\n|\Z)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        evidence = evidence_match.group(1).strip() if evidence_match else "Based on available data."

        return styles, confidence, evidence[:500]  # Limit evidence length

    def _fallback_classification(
        self,
        team_name: str,
        stats: str
    ) -> Tuple[List[str], str, str]:
        """Fallback classification when LLM is unavailable."""
        # Simple heuristic based on common team styles
        known_styles = {
            "Liverpool": ["HIGH_PRESS", "WING_PLAY"],
            "Man City": ["POSSESSION", "TECHNICAL"],
            "Arsenal": ["POSSESSION", "TECHNICAL"],
            "Chelsea": ["POSSESSION", "TECHNICAL"],
            "Tottenham": ["COUNTER_ATTACK", "TECHNICAL"],
            "Man United": ["COUNTER_ATTACK", "WING_PLAY"],
            "Newcastle": ["COUNTER_ATTACK", "PHYSICAL"],
            "Brighton": ["HIGH_PRESS", "TECHNICAL"],
            "Aston Villa": ["COUNTER_ATTACK", "SET_PIECES"],
            "West Ham": ["COUNTER_ATTACK", "PHYSICAL"],
            "Burnley": ["LOW_BLOCK", "PHYSICAL"],
            "Crystal Palace": ["COUNTER_ATTACK", "WING_PLAY"],
            "Wolves": ["COUNTER_ATTACK", "LOW_BLOCK"],
            "Leicester": ["COUNTER_ATTACK"],
            "Everton": ["LOW_BLOCK", "PHYSICAL"],
            "Brentford": ["HIGH_PRESS", "SET_PIECES"],
            "Fulham": ["POSSESSION", "TECHNICAL"],
            "Bournemouth": ["HIGH_PRESS"],
            "Nottingham Forest": ["LOW_BLOCK", "COUNTER_ATTACK"],
            "Leeds United": ["HIGH_PRESS"],
            "Southampton": ["POSSESSION"],
        }

        # Try exact match first
        if team_name in known_styles:
            return known_styles[team_name], "MEDIUM", "Fallback classification (LLM unavailable)."

        # Try partial match
        for known_team, styles in known_styles.items():
            if known_team.lower() in team_name.lower() or team_name.lower() in known_team.lower():
                return styles, "LOW", "Partial match fallback."

        # Default
        return ["POSSESSION"], "LOW", "Default classification (no data)."

    def _save_to_cache(
        self,
        team_name: str,
        styles: List[str],
        confidence: str,
        evidence: str
    ):
        """Save team style to cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO kg_team_styles
                (team_name, styles, confidence, last_updated, evidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                team_name,
                json.dumps(styles),
                confidence,
                datetime.now().isoformat(),
                evidence
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache save error: {e}")

    # =========================================================================
    # Tactical Matchup Analysis
    # =========================================================================

    def get_tactical_matchup(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Analyze tactical matchup between two teams.

        Gets styles for both teams (generating if needed), then analyzes
        tactical advantages based on counter relationships.

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Dict with:
            - home_styles: List of home team's styles
            - away_styles: List of away team's styles
            - advantages: Dict with home/away advantages
            - matchup_summary: String description
            - confidence: Overall confidence level
        """
        # Get styles for both teams
        home_styles = self.get_team_style(home_team)
        away_styles = self.get_team_style(away_team)

        # Analyze matchup
        home_advantages = ["HOME_ADVANTAGE"]  # Always include
        away_advantages = []

        for h_style in home_styles:
            h_counters = self.TACTICAL_COUNTERS.get(h_style, {})

            for a_style in away_styles:
                a_counters = self.TACTICAL_COUNTERS.get(a_style, {})

                # Home effective against away?
                if a_style in h_counters.get("effective_against", []):
                    home_advantages.append(f"{h_style} effective vs {a_style}")

                # Away counters home?
                if h_style in a_counters.get("effective_against", []):
                    away_advantages.append(f"{a_style} effective vs {h_style}")

                # Home counters away?
                if a_style in h_counters.get("countered_by", []):
                    # This means away_style counters home_style
                    pass  # Already covered above

        # Generate summary
        home_tactical = len(home_advantages) - 1  # Exclude home advantage
        away_tactical = len(away_advantages)

        if away_tactical > home_tactical:
            summary = f"Tactical edge to {away_team}. {'; '.join(away_advantages[:2])}"
            confidence = "medium"
        elif home_tactical > away_tactical:
            summary = f"Tactical edge to {home_team}. {'; '.join(home_advantages[1:3])}"
            confidence = "medium"
        else:
            summary = f"Balanced tactical matchup between {home_team} and {away_team}."
            confidence = "low"

        return {
            "home_styles": home_styles,
            "away_styles": away_styles,
            "advantages": {
                "home": home_advantages,
                "away": away_advantages
            },
            "matchup_summary": summary,
            "confidence": confidence
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def refresh_team_style(self, team_name: str) -> List[str]:
        """Force refresh a team's style (ignores cache)."""
        return self.get_team_style(team_name, force_refresh=True)

    def get_cached_teams(self) -> List[str]:
        """Get list of all cached team names."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT team_name FROM kg_team_styles ORDER BY team_name")
            teams = [row[0] for row in cursor.fetchall()]
            conn.close()
            return teams
        except Exception as e:
            logger.error(f"Error getting cached teams: {e}")
            return []

    def get_team_info(self, team_name: str) -> Optional[Dict[str, Any]]:
        """Get full cached info for a team."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT styles, confidence, last_updated, evidence FROM kg_team_styles WHERE team_name = ?",
                (team_name,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "team": team_name,
                    "styles": json.loads(row[0]),
                    "confidence": row[1],
                    "last_updated": row[2],
                    "evidence": row[3]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting team info: {e}")
            return None

    def get_kg_stats(self) -> Dict[str, Any]:
        """Get knowledge graph cache statistics."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Basic counts
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_teams,
                    MIN(last_updated) as oldest_entry,
                    MAX(last_updated) as newest_entry
                FROM kg_team_styles
            """)
            row = cursor.fetchone()

            # Style distribution
            cursor2 = conn.execute("SELECT styles FROM kg_team_styles")
            all_styles = []
            for (styles_json,) in cursor2:
                all_styles.extend(json.loads(styles_json))

            conn.close()

            return {
                "total_teams": row[0],
                "oldest_entry": row[1],
                "newest_entry": row[2],
                "style_distribution": dict(Counter(all_styles))
            }
        except Exception as e:
            logger.error(f"Error getting KG stats: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear all cached team styles."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM kg_team_styles")
            conn.commit()
            conn.close()
            print("  ✓ Knowledge graph cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def visualize_matchup(self, home_team: str, away_team: str):
        """Print formatted tactical matchup analysis."""
        matchup = self.get_tactical_matchup(home_team, away_team)

        print("=" * 70)
        print(f" TACTICAL MATCHUP: {home_team} vs {away_team}")
        print("=" * 70)
        print()
        print(f"🏠 {home_team} (HOME)")
        print(f"   Styles: {', '.join(matchup['home_styles'])}")
        print(f"   Advantages:")
        for adv in matchup['advantages']['home']:
            print(f"      ✓ {adv}")
        print()
        print(f"✈️  {away_team} (AWAY)")
        print(f"   Styles: {', '.join(matchup['away_styles'])}")
        print(f"   Advantages:")
        for adv in matchup['advantages']['away']:
            print(f"      ✓ {adv}")
        if not matchup['advantages']['away']:
            print("      (none)")
        print()
        print(f"📊 SUMMARY: {matchup['matchup_summary']}")
        print(f"   Confidence: {matchup['confidence'].upper()}")
        print("=" * 70)


# =============================================================================
# Tests
# =============================================================================

def test_dynamic_kg():
    """Test dynamic knowledge graph."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    print("\n" + "=" * 70)
    print(" DYNAMIC KNOWLEDGE GRAPH TEST")
    print("=" * 70)

    # Check requirements
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("\n⚠️  TAVILY_API_KEY not set")
        print("   Run: export TAVILY_API_KEY='your-key'")
        print("   Continuing with limited functionality...\n")

    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("✓ Ollama connected")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   Run: ollama serve")

    # Initialize web RAG
    web_rag = None
    if tavily_key:
        try:
            from src.rag.web_search_rag import WebSearchRAG
            web_rag = WebSearchRAG(tavily_api_key=tavily_key)
            print("✓ Tavily web search initialized")
        except Exception as e:
            print(f"⚠️  Web search error: {e}")

    # Create KG
    db_path = str(DEFAULT_DB_PATH)
    kg = DynamicKnowledgeGraph(db_path, web_rag, ollama_model="llama3.2:3b")

    # Clear cache for fresh test
    print("\n--- Clearing cache for fresh test ---")
    kg.clear_cache()

    # Test 1: First call (generation)
    print("\n" + "-" * 50)
    print("TEST 1: Generate style for Liverpool")
    print("-" * 50)
    liverpool_styles = kg.get_team_style("Liverpool")
    print(f"Result: {liverpool_styles}")

    # Test 2: Second call (should use cache)
    print("\n" + "-" * 50)
    print("TEST 2: Get Liverpool again (should use cache)")
    print("-" * 50)
    liverpool_cached = kg.get_team_style("Liverpool")
    print(f"Result: {liverpool_cached}")
    print(f"Cache working: {liverpool_styles == liverpool_cached}")

    # Test 3: Different team
    print("\n" + "-" * 50)
    print("TEST 3: Generate style for Arsenal")
    print("-" * 50)
    arsenal_styles = kg.get_team_style("Arsenal")
    print(f"Result: {arsenal_styles}")

    # Test 4: Tactical matchup
    print("\n" + "-" * 50)
    print("TEST 4: Tactical Matchup Analysis")
    print("-" * 50)
    kg.visualize_matchup("Liverpool", "Arsenal")

    # Test 5: Get team info
    print("\n" + "-" * 50)
    print("TEST 5: Get cached team info")
    print("-" * 50)
    info = kg.get_team_info("Liverpool")
    if info:
        print(f"Team: {info['team']}")
        print(f"Styles: {info['styles']}")
        print(f"Confidence: {info['confidence']}")
        print(f"Evidence: {info['evidence'][:100]}...")
        print(f"Last updated: {info['last_updated']}")

    # Test 6: KG Statistics
    print("\n" + "-" * 50)
    print("TEST 6: Knowledge Graph Statistics")
    print("-" * 50)
    stats = kg.get_kg_stats()
    print(f"Total teams cached: {stats.get('total_teams', 0)}")
    print(f"Style distribution: {stats.get('style_distribution', {})}")

    # Test 7: Force refresh
    print("\n" + "-" * 50)
    print("TEST 7: Force refresh Liverpool")
    print("-" * 50)
    refreshed = kg.refresh_team_style("Liverpool")
    print(f"Refreshed result: {refreshed}")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_dynamic_kg()
