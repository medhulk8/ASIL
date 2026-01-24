"""
Hybrid Agent for Football Match Predictions

This agent integrates multiple data sources and reasoning approaches:
1. MCP Tools - Database access for match stats and form data
2. Knowledge Graph - Tactical style analysis and matchup insights
3. Web Search RAG - Real-time information (injuries, news, etc.)
4. Ollama LLM - Local language model for reasoning

The hybrid approach combines statistical analysis with tactical reasoning
and current context to produce more informed predictions.

Usage:
    python src/agent/hybrid_agent.py

Requirements:
    - Ollama installed and running (ollama serve)
    - Model available (ollama pull llama3.1:8b)
    - Tavily API key set as environment variable
"""

import asyncio
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp" / "sports-lab" / "dist" / "index.js"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "asil.db"

# Add project root to path
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# MCP Client Setup
# ============================================================================

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Wrapper for MCP client session with tool calling."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return parsed JSON response."""
        result = await self.session.call_tool(name, arguments)
        if result.content and len(result.content) > 0:
            content = result.content[0]
            if hasattr(content, 'text'):
                return json.loads(content.text)
        raise RuntimeError(f"Tool '{name}' returned empty response")


@asynccontextmanager
async def connect_to_mcp():
    """Context manager for MCP server connection."""
    server_params = StdioServerParameters(
        command="node",
        args=[str(MCP_SERVER_PATH)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPClient(session)


# ============================================================================
# Hybrid Agent Class
# ============================================================================

class HybridAgent:
    """
    Hybrid prediction agent combining KG + Web Search + Ollama LLM.

    This agent orchestrates multiple data sources:
    - MCP tools for database access (match stats, form, baseline)
    - Knowledge Graph for tactical analysis
    - Web Search RAG for current information
    - Ollama LLM for reasoning and prediction

    Attributes:
        mcp_client: Connected MCP client
        db_path: Path to SQLite database
        kg: FootballKnowledgeGraph instance
        web_rag: WebSearchRAG instance
        ollama: Ollama client
        model: LLM model name
    """

    # Preferred models in order of quality
    PREFERRED_MODELS = ["llama3.1:8b", "llama3.2:3b", "mistral", "phi3"]

    def __init__(
        self,
        mcp_client: MCPClient,
        db_path: Path,
        tavily_api_key: Optional[str] = None,
        kg=None,
        model: str = "llama3.1:8b"
    ):
        """
        Initialize hybrid agent with all components.

        Args:
            mcp_client: MCP client for database tools
            db_path: Path to SQLite database
            tavily_api_key: Tavily API key for web search (optional)
            kg: FootballKnowledgeGraph instance (optional)
            model: Ollama model name (default: llama3.1:8b)
        """
        self.mcp_client = mcp_client
        self.db_path = db_path

        # Initialize Knowledge Graph
        if kg is None:
            from src.kg.football_knowledge_graph import FootballKnowledgeGraph
            self.kg = FootballKnowledgeGraph()
        else:
            self.kg = kg

        # Initialize Web Search RAG (optional)
        self.web_rag = None
        if tavily_api_key:
            try:
                from src.rag.web_search_rag import WebSearchRAG
                self.web_rag = WebSearchRAG(tavily_api_key=tavily_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize web search: {e}")

        # Initialize Ollama with model availability check
        self.ollama = None
        self.model = model
        try:
            import ollama
            self.ollama = ollama

            # Check available models
            models_response = self.ollama.list()
            if hasattr(models_response, 'models'):
                available_models = [m.model for m in models_response.models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                available_models = [m.get('name', m.get('model', '')) for m in models_response['models']]
            else:
                available_models = []

            # Check if requested model is available
            model_found = any(model in m or m.startswith(model.split(':')[0]) for m in available_models)

            if not model_found:
                logger.warning(f"Model '{model}' not found. Available: {available_models}")
                # Try to find a fallback
                fallback = None
                for preferred in self.PREFERRED_MODELS:
                    if any(preferred in m for m in available_models):
                        fallback = preferred
                        break
                if fallback:
                    logger.info(f"Using fallback model: {fallback}")
                    self.model = fallback
                else:
                    logger.error(f"No suitable model found. Install with: ollama pull {model}")
                    print(f"   ⚠️ Model '{model}' not found.")
                    print(f"   Install with: ollama pull {model}")
            else:
                logger.info(f"Ollama connected, using model: {model}")

        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            self.ollama = None

    async def predict_match(
        self,
        match_id: int,
        verbose: bool = True,
        use_web_search: bool = True
    ) -> Dict[str, Any]:
        """
        Complete prediction workflow with hybrid reasoning.

        Steps:
        1. Gather stats context (MCP tools)
        2. Query knowledge graph (tactical insights)
        3. Execute web searches (current info) - optional
        4. Build complete context
        5. LLM reasoning (Ollama)
        6. Self-critique
        7. Log prediction
        8. Evaluate (if match completed)

        Args:
            match_id: ID of match to predict
            verbose: Print detailed output
            use_web_search: Whether to use web search (costs API calls)

        Returns:
            dict with prediction, reasoning, evaluation
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" HYBRID PREDICTION: Match {match_id}")
            print(f"{'='*70}\n")

        # =====================================================================
        # STEP 1: Gather Statistics
        # =====================================================================
        if verbose:
            print("📊 STEP 1: Gathering statistics...")

        match = await self.mcp_client.call_tool("get_match", {"match_id": match_id})

        if "error" in match:
            return {"error": match["error"], "match_id": match_id}

        home_form = await self.mcp_client.call_tool("get_team_form", {
            "team_name": match['home_team'],
            "window": 5
        })

        away_form = await self.mcp_client.call_tool("get_team_form", {
            "team_name": match['away_team'],
            "window": 5
        })

        baseline = await self.mcp_client.call_tool("get_baseline_probs", {
            "match_id": match_id
        })

        if verbose:
            print(f"   Match: {match['home_team']} vs {match['away_team']}")
            print(f"   Date: {match['date']}")
            print(f"   Home form: {home_form.get('form_string', 'N/A')} ({home_form.get('points_per_game', 0):.2f} PPG)")
            print(f"   Away form: {away_form.get('form_string', 'N/A')} ({away_form.get('points_per_game', 0):.2f} PPG)")
            print(f"   Baseline: H={baseline['home_prob']:.1%}, D={baseline['draw_prob']:.1%}, A={baseline['away_prob']:.1%}")

        # =====================================================================
        # STEP 2: Query Knowledge Graph
        # =====================================================================
        if verbose:
            print("\n🧠 STEP 2: Querying Knowledge Graph...")

        kg_insights = self.kg.get_tactical_matchup(
            match['home_team'],
            match['away_team']
        )

        if verbose:
            if kg_insights.get('home_styles'):
                print(f"   Home style: {', '.join(kg_insights['home_styles'])}")
                print(f"   Away style: {', '.join(kg_insights['away_styles'])}")
                print(f"   Matchup: {kg_insights['matchup_summary']}")
                print(f"   Confidence: {kg_insights['confidence']}")
            else:
                print("   No tactical info in KG for these teams")

        # =====================================================================
        # STEP 3: Web Search (Optional)
        # =====================================================================
        rag_context = "No web search performed."

        if use_web_search and self.web_rag:
            if verbose:
                print("\n🌐 STEP 3: Executing web searches...")

            try:
                rag_results = self.web_rag.get_match_context(
                    match['home_team'],
                    match['away_team'],
                    match_date=match['date'],
                    max_searches=3
                )
                rag_context = rag_results.get('all_content', 'No results found.')

                if verbose:
                    print(f"   Queries executed: {rag_results['queries_executed']}")
                    print(f"   Cache hits: {rag_results['cached_hits']}")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
                rag_context = f"Web search unavailable: {str(e)}"
                if verbose:
                    print(f"   ⚠️ Web search failed: {e}")
        elif verbose:
            print("\n🌐 STEP 3: Skipping web search")

        # =====================================================================
        # STEP 4: Build LLM Context
        # =====================================================================
        context_for_llm = self._build_llm_context(
            match, home_form, away_form, baseline, kg_insights, rag_context
        )

        # =====================================================================
        # STEP 5: LLM Reasoning
        # =====================================================================
        if verbose:
            print("\n🤖 STEP 4: LLM Reasoning (Ollama)...")

        prediction = await self._llm_predict(context_for_llm, baseline, verbose)

        if verbose:
            print(f"   Prediction: H={prediction['home_prob']:.1%}, D={prediction['draw_prob']:.1%}, A={prediction['away_prob']:.1%}")
            print(f"   Confidence: {prediction['confidence']}")
            if 'parse_method' in prediction:
                print(f"   Parse method: {prediction['parse_method']}")

        # =====================================================================
        # STEP 6: Self-Critique
        # =====================================================================
        if verbose:
            print("\n🔍 STEP 5: Self-Critique...")

        critique = await self._self_critique(prediction, context_for_llm, verbose)

        # =====================================================================
        # STEP 7: Log Prediction
        # =====================================================================
        if verbose:
            print("\n💾 STEP 6: Logging prediction...")

        log_result = await self.mcp_client.call_tool("log_prediction", {
            "match_id": match_id,
            "baseline_home_prob": baseline['home_prob'],
            "baseline_draw_prob": baseline['draw_prob'],
            "baseline_away_prob": baseline['away_prob'],
            "llm_home_prob": prediction['home_prob'],
            "llm_draw_prob": prediction['draw_prob'],
            "llm_away_prob": prediction['away_prob'],
            "rationale_text": prediction['reasoning'][:1000],  # Truncate if too long
            "timestamp": datetime.now().isoformat()
        })

        if verbose:
            print(f"   ✓ Logged as prediction_id: {log_result['prediction_id']}")

        # =====================================================================
        # STEP 8: Evaluate (if match completed)
        # =====================================================================
        evaluation = None
        if match.get('home_goals') is not None:
            if verbose:
                print("\n📈 STEP 7: Evaluating prediction...")

            evaluation = await self.mcp_client.call_tool("evaluate_prediction", {
                "match_id": match_id
            })

            if verbose:
                print(f"   Outcome: {evaluation['outcome']}")
                print(f"   LLM predicted: {evaluation['llm_predicted']}")
                print(f"   LLM Brier: {evaluation['llm_brier_score']:.4f}")
                print(f"   Correct: {'✓' if evaluation['llm_correct'] else '✗'}")

        if verbose:
            print(f"\n{'='*70}")
            print(" PREDICTION COMPLETE")
            print(f"{'='*70}\n")

        return {
            "match_id": match_id,
            "match": match,
            "prediction": prediction,
            "critique": critique,
            "evaluation": evaluation,
            "prediction_id": log_result['prediction_id'],
            "success": True
        }

    def _build_llm_context(
        self,
        match: Dict,
        home_form: Dict,
        away_form: Dict,
        baseline: Dict,
        kg_insights: Dict,
        rag_context: str
    ) -> str:
        """
        Format all context into a structured prompt for the LLM.

        Returns clean, formatted string with all relevant information.
        """
        kg_text = self._format_kg_insights(kg_insights)

        # Truncate rag_context if too long
        if len(rag_context) > 2000:
            rag_context = rag_context[:2000] + "\n[...truncated for length]"

        context = f"""You are a football match prediction expert. Analyze the following match and predict the outcome probabilities.

MATCH DETAILS:
- Home: {match['home_team']}
- Away: {match['away_team']}
- Date: {match['date']}
- Season: {match.get('season', 'Unknown')}

STATISTICAL CONTEXT:

{match['home_team']} (Home) - Last 5 matches:
- Record: {home_form.get('wins', 0)}W-{home_form.get('draws', 0)}D-{home_form.get('losses', 0)}L
- Points per game: {home_form.get('points_per_game', 0):.2f}
- Goals scored: {home_form.get('goals_scored', 0)}
- Goals conceded: {home_form.get('goals_conceded', 0)}
- Form: {home_form.get('form_string', 'N/A')}

{match['away_team']} (Away) - Last 5 matches:
- Record: {away_form.get('wins', 0)}W-{away_form.get('draws', 0)}D-{away_form.get('losses', 0)}L
- Points per game: {away_form.get('points_per_game', 0):.2f}
- Goals scored: {away_form.get('goals_scored', 0)}
- Goals conceded: {away_form.get('goals_conceded', 0)}
- Form: {away_form.get('form_string', 'N/A')}

BOOKMAKER BASELINE (market consensus):
- Home win: {baseline['home_prob']:.1%}
- Draw: {baseline['draw_prob']:.1%}
- Away win: {baseline['away_prob']:.1%}

TACTICAL ANALYSIS:
{kg_text}

CURRENT INFORMATION:
{rag_context}

TASK:
Analyze all information and provide your prediction. Consider:
1. Recent form and momentum
2. Tactical matchup (does one style counter the other?)
3. Home advantage
4. Any relevant news or injuries

Respond in this EXACT format:
PROBABILITIES: H=XX%, D=XX%, A=XX%
REASONING: [Your 2-3 sentence analysis]
CONFIDENCE: [HIGH/MEDIUM/LOW]
"""
        return context

    def _format_kg_insights(self, kg_insights: Dict) -> str:
        """Format KG insights for LLM prompt."""
        if not kg_insights.get('home_styles'):
            return "No tactical information available for these teams."

        lines = [
            f"Home team plays: {', '.join(kg_insights['home_styles'])}",
            f"Away team plays: {', '.join(kg_insights['away_styles'])}",
            f"Matchup analysis: {kg_insights['matchup_summary']}",
            f"Tactical confidence: {kg_insights['confidence']}"
        ]

        # Add advantages
        home_advs = kg_insights.get('advantages', {}).get('home', [])
        away_advs = kg_insights.get('advantages', {}).get('away', [])

        if home_advs:
            lines.append(f"Home advantages: {', '.join(home_advs[:2])}")
        if away_advs:
            lines.append(f"Away advantages: {', '.join(away_advs[:2])}")

        return "\n".join(lines)

    async def _llm_predict(
        self,
        context: str,
        baseline: Dict,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Call Ollama LLM to make prediction.

        Returns:
            dict with home_prob, draw_prob, away_prob, reasoning, confidence
        """
        if not self.ollama:
            if verbose:
                print("   ⚠️ Ollama not available, using baseline")
            return {
                "home_prob": baseline['home_prob'],
                "draw_prob": baseline['draw_prob'],
                "away_prob": baseline['away_prob'],
                "reasoning": "LLM unavailable, using baseline probabilities.",
                "confidence": "low"
            }

        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": context
                }],
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent output
                    "num_predict": 500   # Limit response length
                }
            )

            response_text = response['message']['content']

            if verbose:
                print(f"   Model: {self.model}")
                print(f"   Response length: {len(response_text)} chars")

            # Parse LLM response
            parsed = self._parse_llm_response(response_text, baseline)

            return parsed

        except Exception as e:
            logger.error(f"LLM prediction error: {e}")
            if verbose:
                print(f"   ⚠️ LLM error: {e}")
                print("   Falling back to baseline probabilities")

            return {
                "home_prob": baseline['home_prob'],
                "draw_prob": baseline['draw_prob'],
                "away_prob": baseline['away_prob'],
                "reasoning": f"LLM error: {str(e)}. Using baseline.",
                "confidence": "low"
            }

    def _parse_llm_response(self, response_text: str, baseline: Dict) -> Dict[str, Any]:
        """
        Parse LLM response to extract probabilities and reasoning.

        Uses multiple parsing strategies in order of preference:
        1. Exact format: "H=X%, D=Y%, A=Z%"
        2. Keyword format: "Home: X%, Draw: Y%, Away: Z%"
        3. Labeled format: "Home win: X%", "Draw: Y%", "Away win: Z%"
        4. Any 3 consecutive percentages
        5. Fallback to baseline

        Returns:
            Dict with probabilities, reasoning, confidence, and parsing info
        """
        home_prob = None
        draw_prob = None
        away_prob = None
        parse_method = None

        # Strategy 1: Exact format "H=X%, D=Y%, A=Z%"
        match1 = re.search(
            r'H\s*=\s*(\d+(?:\.\d+)?)\s*%.*?D\s*=\s*(\d+(?:\.\d+)?)\s*%.*?A\s*=\s*(\d+(?:\.\d+)?)\s*%',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if match1:
            home_prob = float(match1.group(1)) / 100
            draw_prob = float(match1.group(2)) / 100
            away_prob = float(match1.group(3)) / 100
            parse_method = "exact_format"

        # Strategy 2: Keyword format "Home: X%, Draw: Y%, Away: Z%"
        if home_prob is None:
            match2 = re.search(
                r'Home[:\s]+(\d+(?:\.\d+)?)\s*%.*?Draw[:\s]+(\d+(?:\.\d+)?)\s*%.*?Away[:\s]+(\d+(?:\.\d+)?)\s*%',
                response_text,
                re.IGNORECASE | re.DOTALL
            )
            if match2:
                home_prob = float(match2.group(1)) / 100
                draw_prob = float(match2.group(2)) / 100
                away_prob = float(match2.group(3)) / 100
                parse_method = "keyword_format"

        # Strategy 3: Labeled format with "win" keyword
        if home_prob is None:
            home_match = re.search(r'Home\s*(?:win)?[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)
            draw_match = re.search(r'Draw[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)
            away_match = re.search(r'Away\s*(?:win)?[:\s]+(\d+(?:\.\d+)?)\s*%', response_text, re.IGNORECASE)

            if home_match and draw_match and away_match:
                home_prob = float(home_match.group(1)) / 100
                draw_prob = float(draw_match.group(1)) / 100
                away_prob = float(away_match.group(1)) / 100
                parse_method = "labeled_format"

        # Strategy 4: Look for bullet points or list format
        if home_prob is None:
            # Match patterns like "- Home win: 45%" or "* Home: 45%"
            bullet_pattern = re.findall(
                r'[-*•]\s*(?:Home|Draw|Away)[^:]*:\s*(\d+(?:\.\d+)?)\s*%',
                response_text,
                re.IGNORECASE
            )
            if len(bullet_pattern) >= 3:
                home_prob = float(bullet_pattern[0]) / 100
                draw_prob = float(bullet_pattern[1]) / 100
                away_prob = float(bullet_pattern[2]) / 100
                parse_method = "bullet_format"

        # Strategy 5: Any 3 consecutive percentages (last resort before baseline)
        if home_prob is None:
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', response_text)
            if len(percentages) >= 3:
                # Take first 3 that look like valid probabilities (0-100)
                valid_pcts = [float(p) for p in percentages if 0 <= float(p) <= 100]
                if len(valid_pcts) >= 3:
                    home_prob = valid_pcts[0] / 100
                    draw_prob = valid_pcts[1] / 100
                    away_prob = valid_pcts[2] / 100
                    parse_method = "any_percentages"

        # Strategy 6: Fallback to baseline
        if home_prob is None:
            home_prob = baseline['home_prob']
            draw_prob = baseline['draw_prob']
            away_prob = baseline['away_prob']
            parse_method = "baseline_fallback"
            logger.warning("Could not parse LLM response, using baseline probabilities")

        # Validate and normalize probabilities
        # Ensure they're in valid range
        home_prob = max(0.01, min(0.98, home_prob))
        draw_prob = max(0.01, min(0.98, draw_prob))
        away_prob = max(0.01, min(0.98, away_prob))

        # Normalize to sum to 1.0
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        else:
            home_prob, draw_prob, away_prob = 0.4, 0.3, 0.3
            parse_method = "normalization_fallback"

        # Extract reasoning
        reasoning = ""
        reasoning_patterns = [
            r'REASONING:?\s*(.+?)(?:CONFIDENCE|$)',
            r'Analysis:?\s*(.+?)(?:CONFIDENCE|PROBABILITIES|$)',
            r'(?:My |The )?(?:analysis|reasoning|prediction)[:\s]+(.+?)(?:CONFIDENCE|PROBABILITIES|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                break

        if not reasoning:
            # Use full response, cleaned up
            reasoning = response_text[:500].strip()

        # Extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:?\s*(HIGH|MEDIUM|LOW)',
            response_text,
            re.IGNORECASE
        )
        confidence = confidence_match.group(1).lower() if confidence_match else "medium"

        # Log parsing method for debugging
        logger.debug(f"Parsed using: {parse_method}")

        return {
            "home_prob": round(home_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "away_prob": round(away_prob, 4),
            "reasoning": reasoning,
            "confidence": confidence,
            "parse_method": parse_method,
            "raw_response": response_text
        }

    async def _self_critique(
        self,
        prediction: Dict,
        context: str,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        LLM critiques its own prediction.

        Returns:
            dict with critique text and warnings
        """
        if not self.ollama:
            return {"critique": "Self-critique unavailable (no LLM)", "warnings": []}

        critique_prompt = f"""You just predicted this football match:
- Home win: {prediction['home_prob']:.1%}
- Draw: {prediction['draw_prob']:.1%}
- Away win: {prediction['away_prob']:.1%}
- Confidence: {prediction['confidence']}

Brief reasoning: {prediction['reasoning'][:300]}

Now critique your prediction in 1-2 sentences. Are there any concerns or factors you might have missed?"""

        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": critique_prompt}],
                options={"temperature": 0.5, "num_predict": 200}
            )

            critique_text = response['message']['content']

            if verbose:
                print(f"   {critique_text[:200]}...")

            return {
                "critique": critique_text,
                "warnings": []
            }

        except Exception as e:
            return {
                "critique": f"Self-critique unavailable: {str(e)}",
                "warnings": []
            }


# ============================================================================
# Main Function
# ============================================================================

async def main():
    """
    Test hybrid agent with example matches.
    """
    print("=" * 70)
    print(" HYBRID AGENT TEST")
    print("=" * 70)

    # Check Ollama
    print("\n1. Checking Ollama...")
    try:
        import ollama
        models = ollama.list()
        print(f"   ✓ Ollama connected")
        # Handle different response formats
        if hasattr(models, 'models'):
            model_names = [m.model for m in models.models]
        elif isinstance(models, dict) and 'models' in models:
            model_names = [m.get('name', m.get('model', 'unknown')) for m in models['models']]
        else:
            model_names = ['llama3.1:8b']
        print(f"   Available models: {model_names}")
    except Exception as e:
        print(f"   ✗ Ollama error: {e}")
        print("\n   To fix:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Start service: ollama serve")
        print("   3. Pull a model: ollama pull llama3.1:8b")
        return

    # Check Tavily
    print("\n2. Checking Tavily API key...")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        print(f"   ✓ TAVILY_API_KEY found: {tavily_key[:10]}...")
    else:
        print("   ⚠️ TAVILY_API_KEY not set (web search disabled)")
        print("   To enable: export TAVILY_API_KEY='your-key'")

    # Initialize
    print("\n3. Connecting to MCP server...")
    async with connect_to_mcp() as mcp_client:
        print("   ✓ MCP connected")

        # Create agent
        print("\n4. Initializing Hybrid Agent...")
        agent = HybridAgent(
            mcp_client=mcp_client,
            db_path=DEFAULT_DB_PATH,
            tavily_api_key=tavily_key,
            model="llama3.1:8b"
        )
        print("   ✓ Agent initialized")

        # Test predictions
        print("\n" + "=" * 70)
        print(" RUNNING PREDICTIONS")
        print("=" * 70)

        test_matches = [1, 50]  # Test with 2 matches

        results = []
        for match_id in test_matches:
            try:
                result = await agent.predict_match(
                    match_id,
                    verbose=True,
                    use_web_search=bool(tavily_key)
                )
                results.append(result)

                if result.get('evaluation'):
                    status = "✓" if result['evaluation']['llm_correct'] else "✗"
                    print(f"\nMatch {match_id}: {status}")
                else:
                    print(f"\nMatch {match_id}: Prediction logged")

            except Exception as e:
                logger.error(f"Error predicting match {match_id}: {e}")
                print(f"\n✗ Error predicting match {match_id}: {e}")

        # Summary
        print("\n" + "=" * 70)
        print(" SUMMARY")
        print("=" * 70)

        correct = sum(1 for r in results if r.get('evaluation', {}).get('llm_correct'))
        evaluated = sum(1 for r in results if r.get('evaluation'))

        print(f"\nPredictions made: {len(results)}")
        print(f"Evaluated: {evaluated}")
        print(f"Correct: {correct}")
        if evaluated > 0:
            print(f"Accuracy: {correct/evaluated:.1%}")

        print("\n" + "=" * 70)
        print(" TEST COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
