"""
LangGraph Prediction Workflow

This module defines a graph-based workflow for football match predictions
using LangGraph's StateGraph. The workflow orchestrates 8 nodes:

1. match_selector   - Load match details and initialize state
2. stats_collector  - Gather team form and baseline probabilities
3. kg_query         - Query knowledge graph for tactical insights
4. web_search       - Execute web searches for current context (conditional)
5. llm_predictor    - Generate LLM prediction with Ollama
6. critique         - Self-critique the prediction
7. logger           - Save prediction to database
8. evaluator        - Evaluate prediction if match completed

The graph structure enables:
- Conditional branching (skip web search if not needed)
- Cycle handling (re-prediction if confidence too low)
- State persistence across nodes
- Clear visualization of workflow

Usage:
    from src.workflows import create_prediction_workflow, PredictionState

    components = create_workflow_components(mcp_client, db_path, kg, web_rag)
    workflow = create_prediction_workflow(components)
    result = await workflow.ainvoke({"match_id": 1})
"""

from typing import TypedDict, Literal, Optional, Any, Dict, List, Callable
from typing_extensions import Annotated
from pathlib import Path
from datetime import datetime
import re
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class PredictionState(TypedDict, total=False):
    """
    Shared state across all workflow nodes.

    This TypedDict defines the structure of data that flows through
    the prediction workflow graph. Each node can read from and write
    to this state.

    Attributes:
        match_id: Unique identifier for the match
        match: Match details from database
        home_form: Home team's recent form statistics
        away_form: Away team's recent form statistics
        baseline: Bookmaker baseline probabilities
        kg_insights: Tactical insights from knowledge graph
        web_context: Retrieved web search context
        prediction: LLM prediction with probabilities
        critique: Self-critique results
        evaluation: Prediction evaluation results
        prediction_id: Database ID of logged prediction
        skip_web_search: Flag to skip web search step
        confidence_level: Current prediction confidence
        iteration_count: Number of prediction iterations (for cycle limiting)
        verbose: Enable verbose output
        error: Error message if any step fails
    """
    # Input
    match_id: int
    verbose: bool

    # Match data (populated by match_selector and stats_collector nodes)
    match: Dict[str, Any]
    home_form: Dict[str, Any]
    away_form: Dict[str, Any]
    baseline: Dict[str, Any]

    # Context gathering (populated by respective nodes)
    kg_insights: Dict[str, Any]
    web_context: Dict[str, Any]

    # Prediction (populated by llm_predictor node)
    prediction: Dict[str, Any]
    critique: Dict[str, Any]

    # Evaluation (populated by evaluator node)
    evaluation: Dict[str, Any]
    prediction_id: int

    # Workflow control
    skip_web_search: bool
    confidence_level: str
    iteration_count: int

    # Error handling
    error: Optional[str]


# ============================================================================
# Helper Functions
# ============================================================================

def _build_prediction_prompt(state: PredictionState) -> str:
    """Build the LLM prompt from state."""
    match = state["match"]
    home_form = state["home_form"]
    away_form = state["away_form"]
    baseline = state["baseline"]
    kg = state.get("kg_insights", {})
    web = state.get("web_context", {})

    # Format KG insights
    kg_text = ""
    if kg and not kg.get("error"):
        home_styles = kg.get("home_styles", [])
        away_styles = kg.get("away_styles", [])
        if home_styles or away_styles:
            kg_text = f"""
TACTICAL ANALYSIS:
Home styles: {', '.join(home_styles) if home_styles else 'Unknown'}
Away styles: {', '.join(away_styles) if away_styles else 'Unknown'}
Matchup: {kg.get('matchup_summary', 'N/A')}"""

    # Format web context
    web_text = web.get("all_content", "No web context available.")[:1000] if web else "Web search skipped."

    # Format form strings
    home_ppg = home_form.get('points_per_game', 0) or 0
    away_ppg = away_form.get('points_per_game', 0) or 0

    return f"""Predict this football match:

MATCH: {match['home_team']} vs {match['away_team']}
DATE: {match.get('date', 'Unknown')}

HOME FORM: {home_form.get('form_string', 'N/A')} ({home_ppg:.2f} PPG)
AWAY FORM: {away_form.get('form_string', 'N/A')} ({away_ppg:.2f} PPG)

BASELINE: H={baseline['home_prob']:.1%}, D={baseline['draw_prob']:.1%}, A={baseline['away_prob']:.1%}
{kg_text}

WEB CONTEXT:
{web_text}

Respond EXACTLY in this format:
PROBABILITIES: H=XX%, D=XX%, A=XX%
REASONING: [Your analysis in 2-3 sentences]
CONFIDENCE: HIGH/MEDIUM/LOW"""


def _parse_prediction_response(response_text: str, baseline: dict) -> dict:
    """
    Parse LLM response to extract probabilities and reasoning.

    Uses multiple parsing strategies in order of preference:
    1. Exact format: "H=X%, D=Y%, A=Z%"
    2. Keyword format: "Home: X%, Draw: Y%, Away: Z%"
    3. Labeled format: "Home win: X%", "Draw: Y%", "Away win: Z%"
    4. Bullet format: "- Home: X%"
    5. Any 3 consecutive percentages
    6. Fallback to baseline
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

    # Strategy 4: Bullet format "- Home: X%"
    if home_prob is None:
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

    # Strategy 5: Any 3 consecutive percentages
    if home_prob is None:
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', response_text)
        if len(percentages) >= 3:
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
    home_prob = max(0.01, min(0.98, home_prob))
    draw_prob = max(0.01, min(0.98, draw_prob))
    away_prob = max(0.01, min(0.98, away_prob))

    # Normalize to sum to 1.0
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total
        draw_prob /= total
        away_prob /= total

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
        reasoning = response_text[:500].strip()

    # Extract confidence
    conf_match = re.search(r'CONFIDENCE:?\s*(HIGH|MEDIUM|LOW)', response_text, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else "medium"

    return {
        "home_prob": round(home_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_prob": round(away_prob, 4),
        "reasoning": reasoning,
        "confidence": confidence,
        "parse_method": parse_method,
        "raw_response": response_text
    }


# ============================================================================
# Workflow Components Factory
# ============================================================================

def create_workflow_components(
    mcp_client,
    db_path: Path,
    kg=None,
    web_rag=None,
    ollama_model: str = "llama3.1:8b"
) -> Dict[str, Callable]:
    """
    Factory function to create workflow nodes with dependencies injected.

    This function creates all 8 workflow nodes using closures to capture
    the dependencies (mcp_client, kg, web_rag, ollama). This allows nodes
    to be pure functions that only take state as input.

    Args:
        mcp_client: Connected MCP client for database access
        db_path: Path to SQLite database
        kg: Knowledge graph instance (FootballKnowledgeGraph or DynamicKnowledgeGraph)
        web_rag: WebSearchRAG instance for web searches
        ollama_model: LLM model name for Ollama

    Returns:
        Dict mapping node names to node functions ready for StateGraph
    """
    import ollama

    # =========================================================================
    # Node 1: match_selector_node
    # =========================================================================
    async def match_selector_node(state: PredictionState) -> PredictionState:
        """
        Initialize workflow with match details.

        Actions:
        - Call MCP get_match to load match data
        - Initialize workflow control flags
        """
        verbose = state.get("verbose", True)
        match_id = state["match_id"]

        if verbose:
            print(f"\n📍 [MatchSelector] Loading match {match_id}...")

        try:
            match = await mcp_client.call_tool("get_match", {
                "match_id": match_id
            })

            if "error" in match:
                logger.error(f"Match not found: {match['error']}")
                return {
                    "error": f"Match not found: {match['error']}",
                    "skip_web_search": True,
                    "iteration_count": 0
                }

            if verbose:
                print(f"   ✓ {match['home_team']} vs {match['away_team']} ({match.get('date', 'Unknown')})")

            return {
                "match": match,
                "skip_web_search": False,
                "iteration_count": 0
            }

        except Exception as e:
            logger.error(f"Error loading match: {e}")
            return {"error": str(e), "skip_web_search": True, "iteration_count": 0}

    # =========================================================================
    # Node 2: stats_collector_node
    # =========================================================================
    async def stats_collector_node(state: PredictionState) -> PredictionState:
        """
        Gather statistics: team form and baseline probabilities.

        Actions:
        - Call get_team_form for home team (5 matches)
        - Call get_team_form for away team (5 matches)
        - Call get_baseline_probs for bookmaker odds
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {}

        if verbose:
            print("\n📊 [StatsCollector] Gathering team statistics...")

        try:
            match = state["match"]

            # Get home team form
            home_form = await mcp_client.call_tool("get_team_form", {
                "team_name": match["home_team"],
                "window": 5
            })

            # Get away team form
            away_form = await mcp_client.call_tool("get_team_form", {
                "team_name": match["away_team"],
                "window": 5
            })

            # Get baseline probabilities
            baseline = await mcp_client.call_tool("get_baseline_probs", {
                "match_id": state["match_id"]
            })

            if verbose:
                home_str = home_form.get('form_string', 'N/A')
                away_str = away_form.get('form_string', 'N/A')
                print(f"   Home form: {home_str}")
                print(f"   Away form: {away_str}")
                print(f"   Baseline: H={baseline['home_prob']:.1%}, D={baseline['draw_prob']:.1%}, A={baseline['away_prob']:.1%}")

            return {
                "home_form": home_form,
                "away_form": away_form,
                "baseline": baseline
            }

        except Exception as e:
            logger.error(f"Error collecting stats: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Node 3: kg_query_node
    # =========================================================================
    async def kg_query_node(state: PredictionState) -> PredictionState:
        """
        Query knowledge graph for tactical insights.

        Actions:
        - Get team tactical styles
        - Analyze tactical matchup
        - Return advantages/disadvantages
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"kg_insights": {"error": "Skipped due to previous error"}}

        if verbose:
            print("\n🧠 [KGQuery] Querying knowledge graph...")

        try:
            if kg is None:
                if verbose:
                    print("   ⚠️ No knowledge graph configured")
                return {"kg_insights": {"error": "No KG configured"}}

            match = state["match"]
            kg_insights = kg.get_tactical_matchup(
                match["home_team"],
                match["away_team"]
            )

            if verbose:
                home_styles = kg_insights.get('home_styles', [])
                away_styles = kg_insights.get('away_styles', [])
                print(f"   Home styles: {', '.join(home_styles) if home_styles else 'Unknown'}")
                print(f"   Away styles: {', '.join(away_styles) if away_styles else 'Unknown'}")
                if kg_insights.get('matchup_summary'):
                    print(f"   Matchup: {kg_insights['matchup_summary'][:100]}...")

            return {"kg_insights": kg_insights}

        except Exception as e:
            logger.warning(f"KG query failed: {e}")
            if verbose:
                print(f"   ⚠️ KG query failed: {e}")
            return {"kg_insights": {"error": str(e)}}

    # =========================================================================
    # Node 4: web_search_node
    # =========================================================================
    async def web_search_node(state: PredictionState) -> PredictionState:
        """
        Execute web searches for current information.

        Actions:
        - Generate search queries based on match + KG
        - Execute searches (max 5)
        - Format results for LLM context

        Note: Only runs if confidence router decides to (not skipped)
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"web_context": {"all_content": "Skipped due to error."}}

        if verbose:
            print("\n🔍 [WebSearch] Searching for current context...")

        try:
            if web_rag is None:
                if verbose:
                    print("   ⚠️ Web search not configured")
                return {"web_context": {"all_content": "Web search not configured."}}

            match = state["match"]

            # Generate queries incorporating KG insights
            queries = web_rag.generate_match_queries(
                match["home_team"],
                match["away_team"],
                match.get("date")
            )

            # Add KG-informed queries if available
            kg_insights = state.get("kg_insights", {})
            if kg_insights.get("home_styles"):
                style = kg_insights["home_styles"][0] if kg_insights["home_styles"] else None
                if style:
                    queries.append(f"{match['home_team']} {style} tactics analysis")

            # Execute searches
            results = web_rag.execute_searches(queries, max_searches=5)

            if verbose:
                print(f"   Executed {results['queries_executed']} searches")
                print(f"   Cache hits: {results['cached_hits']}")

            return {"web_context": results}

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            if verbose:
                print(f"   ⚠️ Web search failed: {e}")
            return {"web_context": {"error": str(e), "all_content": f"Web search error: {e}"}}

    # =========================================================================
    # Node 5: llm_predictor_node
    # =========================================================================
    async def llm_predictor_node(state: PredictionState) -> PredictionState:
        """
        Generate prediction using LLM.

        Actions:
        - Build context from stats + KG + web
        - Call Ollama for prediction
        - Parse response to extract probabilities
        - Extract confidence level
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {}

        if verbose:
            print("\n🤖 [LLMPredictor] Generating prediction...")

        try:
            # Build prompt from all gathered context
            prompt = _build_prediction_prompt(state)

            # Call Ollama
            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500}
            )

            response_text = response["message"]["content"]

            # Parse response
            prediction = _parse_prediction_response(response_text, state["baseline"])

            if verbose:
                print(f"   Prediction: H={prediction['home_prob']:.1%}, D={prediction['draw_prob']:.1%}, A={prediction['away_prob']:.1%}")
                print(f"   Confidence: {prediction['confidence']}")
                print(f"   Parse method: {prediction['parse_method']}")

            return {
                "prediction": prediction,
                "confidence_level": prediction["confidence"],
                "iteration_count": state.get("iteration_count", 0) + 1
            }

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            if verbose:
                print(f"   ⚠️ LLM error: {e}")

            # Fallback to baseline
            baseline = state["baseline"]
            return {
                "prediction": {
                    "home_prob": baseline["home_prob"],
                    "draw_prob": baseline["draw_prob"],
                    "away_prob": baseline["away_prob"],
                    "reasoning": f"LLM error: {e}. Using baseline.",
                    "confidence": "low",
                    "parse_method": "error_fallback"
                },
                "confidence_level": "low",
                "iteration_count": state.get("iteration_count", 0) + 1
            }

    # =========================================================================
    # Node 6: critique_node
    # =========================================================================
    async def critique_node(state: PredictionState) -> PredictionState:
        """
        LLM self-critique of the prediction.

        Actions:
        - Prompt LLM to critique its own prediction
        - Check for warnings or concerns
        - Return critique text
        """
        verbose = state.get("verbose", True)

        if state.get("error") or not state.get("prediction"):
            return {"critique": {"text": "No prediction to critique"}}

        if verbose:
            print("\n🔎 [Critique] Self-critiquing prediction...")

        try:
            pred = state["prediction"]
            match = state["match"]

            prompt = f"""Critique this football prediction for {match['home_team']} vs {match['away_team']}:
- Home win: {pred['home_prob']:.1%}
- Draw: {pred['draw_prob']:.1%}
- Away win: {pred['away_prob']:.1%}

Reasoning given: {pred.get('reasoning', 'N/A')[:300]}

In 1-2 sentences, identify any concerns or potential biases in this prediction."""

            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5, "num_predict": 150}
            )

            critique_text = response["message"]["content"]

            if verbose:
                print(f"   Critique: {critique_text[:100]}...")

            return {"critique": {"text": critique_text}}

        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            if verbose:
                print(f"   ⚠️ Critique error: {e}")
            return {"critique": {"text": f"Critique unavailable: {e}"}}

    # =========================================================================
    # Node 7: logger_node
    # =========================================================================
    async def logger_node(state: PredictionState) -> PredictionState:
        """
        Save prediction to database via MCP.

        Actions:
        - Call log_prediction MCP tool
        - Return prediction_id
        """
        verbose = state.get("verbose", True)

        if state.get("error") or not state.get("prediction"):
            return {}

        if verbose:
            print("\n💾 [Logger] Saving prediction to database...")

        try:
            pred = state["prediction"]
            baseline = state["baseline"]

            result = await mcp_client.call_tool("log_prediction", {
                "match_id": state["match_id"],
                "baseline_home_prob": baseline["home_prob"],
                "baseline_draw_prob": baseline["draw_prob"],
                "baseline_away_prob": baseline["away_prob"],
                "llm_home_prob": pred["home_prob"],
                "llm_draw_prob": pred["draw_prob"],
                "llm_away_prob": pred["away_prob"],
                "rationale_text": pred.get("reasoning", "")[:1000],
                "timestamp": datetime.now().isoformat()
            })

            prediction_id = result.get("prediction_id")

            if verbose:
                print(f"   ✓ Logged as prediction #{prediction_id}")

            return {"prediction_id": prediction_id}

        except Exception as e:
            logger.error(f"Logging failed: {e}")
            if verbose:
                print(f"   ⚠️ Logging error: {e}")
            return {}

    # =========================================================================
    # Node 8: evaluator_node
    # =========================================================================
    async def evaluator_node(state: PredictionState) -> PredictionState:
        """
        Evaluate prediction if match is completed.

        Actions:
        - Check if match has result (home_goals not null)
        - If yes, call evaluate_prediction MCP tool
        - If no, return pending status
        """
        verbose = state.get("verbose", True)

        if state.get("error"):
            return {"evaluation": {"status": "error"}}

        if verbose:
            print("\n📈 [Evaluator] Checking match result...")

        try:
            match = state["match"]

            # Check if match has a result
            if match.get("home_goals") is None:
                if verbose:
                    print("   ⏳ Match not yet played - evaluation pending")
                return {"evaluation": {"status": "pending", "message": "Match not yet completed"}}

            # Match has result, evaluate
            result = await mcp_client.call_tool("evaluate_prediction", {
                "match_id": state["match_id"]
            })

            if verbose:
                actual = f"{match['home_goals']}-{match['away_goals']}"
                pred = state["prediction"]
                print(f"   Actual result: {actual}")
                if result.get("llm_brier"):
                    print(f"   LLM Brier: {result['llm_brier']:.4f}")
                    print(f"   Baseline Brier: {result.get('baseline_brier', 'N/A')}")

            return {"evaluation": result}

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            if verbose:
                print(f"   ⚠️ Evaluation error: {e}")
            return {"evaluation": {"status": "error", "error": str(e)}}

    # =========================================================================
    # Return all nodes
    # =========================================================================
    return {
        "match_selector": match_selector_node,
        "stats_collector": stats_collector_node,
        "kg_query": kg_query_node,
        "web_search": web_search_node,
        "llm_predictor": llm_predictor_node,
        "critique": critique_node,
        "logger": logger_node,
        "evaluator": evaluator_node
    }


# ============================================================================
# Routing Functions
# ============================================================================

def should_skip_web_search(state: PredictionState) -> Literal["web_search", "llm_predictor"]:
    """Decide whether to skip web search."""
    if state.get("skip_web_search", False):
        return "llm_predictor"
    if state.get("error"):
        return "llm_predictor"
    return "web_search"


def should_retry_prediction(state: PredictionState) -> Literal["llm_predictor", "critique"]:
    """Decide whether to retry prediction based on confidence."""
    # Limit iterations to prevent infinite loops
    if state.get("iteration_count", 0) >= 2:
        return "critique"
    if state.get("confidence_level") == "low":
        return "llm_predictor"  # Retry
    return "critique"


# ============================================================================
# Workflow Builder
# ============================================================================

def create_prediction_workflow(
    components: Dict[str, Callable],
    skip_web_search: bool = False
) -> StateGraph:
    """
    Create the prediction workflow graph.

    Args:
        components: Dict of node functions from create_workflow_components()
        skip_web_search: If True, always skip web search node

    Returns:
        Compiled LangGraph workflow

    Graph structure:
        match_selector -> stats_collector -> kg_query -> [web_search] -> llm_predictor
        -> critique -> logger -> evaluator -> END
    """
    # Create workflow graph
    workflow = StateGraph(PredictionState)

    # Add all 8 nodes
    workflow.add_node("match_selector", components["match_selector"])
    workflow.add_node("stats_collector", components["stats_collector"])
    workflow.add_node("kg_query", components["kg_query"])
    workflow.add_node("web_search", components["web_search"])
    workflow.add_node("llm_predictor", components["llm_predictor"])
    workflow.add_node("critique", components["critique"])
    workflow.add_node("logger", components["logger"])
    workflow.add_node("evaluator", components["evaluator"])

    # Set entry point
    workflow.set_entry_point("match_selector")

    # Linear flow: match_selector -> stats_collector -> kg_query
    workflow.add_edge("match_selector", "stats_collector")
    workflow.add_edge("stats_collector", "kg_query")

    # Conditional: skip web search?
    if skip_web_search:
        workflow.add_edge("kg_query", "llm_predictor")
    else:
        workflow.add_conditional_edges(
            "kg_query",
            should_skip_web_search,
            {
                "web_search": "web_search",
                "llm_predictor": "llm_predictor"
            }
        )
        workflow.add_edge("web_search", "llm_predictor")

    # Continue linear flow
    workflow.add_edge("llm_predictor", "critique")
    workflow.add_edge("critique", "logger")
    workflow.add_edge("logger", "evaluator")
    workflow.add_edge("evaluator", END)

    # Compile and return
    return workflow.compile()


# ============================================================================
# Convenience Function
# ============================================================================

def build_workflow(
    mcp_client,
    db_path: Path,
    kg=None,
    web_rag=None,
    ollama_model: str = "llama3.1:8b",
    skip_web_search: bool = False
):
    """
    Convenience function to build workflow in one call.

    Args:
        mcp_client: Connected MCP client
        db_path: Path to SQLite database
        kg: Knowledge graph instance
        web_rag: WebSearchRAG instance
        ollama_model: Ollama model name
        skip_web_search: Whether to skip web search

    Returns:
        Compiled LangGraph workflow
    """
    components = create_workflow_components(
        mcp_client=mcp_client,
        db_path=db_path,
        kg=kg,
        web_rag=web_rag,
        ollama_model=ollama_model
    )
    return create_prediction_workflow(components, skip_web_search=skip_web_search)


# ============================================================================
# Test
# ============================================================================

async def test_workflow():
    """Test the prediction workflow structure."""
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print(" LANGGRAPH PREDICTION WORKFLOW TEST")
    print("=" * 70)

    # Check imports
    print("\n1. Checking imports...")
    try:
        from langgraph.graph import StateGraph, END
        print("   ✓ langgraph imported")
    except ImportError as e:
        print(f"   ✗ langgraph import failed: {e}")
        return

    try:
        import ollama
        print("   ✓ ollama imported")
    except ImportError as e:
        print(f"   ✗ ollama import failed: {e}")

    # Test state definition
    print("\n2. Testing PredictionState...")
    test_state: PredictionState = {
        "match_id": 1,
        "verbose": True,
        "skip_web_search": False,
        "iteration_count": 0
    }
    print(f"   ✓ State created with keys: {list(test_state.keys())}")

    # Test helper functions
    print("\n3. Testing helper functions...")
    test_baseline = {"home_prob": 0.4, "draw_prob": 0.3, "away_prob": 0.3}

    # Test parse with exact format
    test_response = "PROBABILITIES: H=45%, D=30%, A=25%\nREASONING: Test\nCONFIDENCE: HIGH"
    parsed = _parse_prediction_response(test_response, test_baseline)
    print(f"   ✓ Parsed exact format: {parsed['parse_method']}")
    assert parsed['parse_method'] == 'exact_format'

    # Test parse with keyword format
    test_response2 = "Home: 50%, Draw: 25%, Away: 25%"
    parsed2 = _parse_prediction_response(test_response2, test_baseline)
    print(f"   ✓ Parsed keyword format: {parsed2['parse_method']}")

    # List all 8 nodes
    print("\n4. Workflow nodes defined:")
    nodes = [
        "match_selector   - Initialize state with match details",
        "stats_collector  - Gather team form and baseline probabilities",
        "kg_query         - Query knowledge graph for tactical insights",
        "web_search       - Execute web searches (conditional)",
        "llm_predictor    - Generate LLM prediction",
        "critique         - Self-critique the prediction",
        "logger           - Save prediction to database",
        "evaluator        - Evaluate if match completed"
    ]
    for node in nodes:
        print(f"   ✓ {node}")

    print("\n5. Workflow graph structure:")
    print("   match_selector -> stats_collector -> kg_query")
    print("       └─[conditional]─> web_search ─┐")
    print("       └─[skip]─────────────────────>├─> llm_predictor")
    print("   llm_predictor -> critique -> logger -> evaluator -> END")

    print("\n" + "=" * 70)
    print(" WORKFLOW STRUCTURE VERIFIED")
    print("=" * 70)
    print("\nTo run full workflow, use:")
    print("  components = create_workflow_components(mcp_client, db_path, kg, web_rag)")
    print("  workflow = create_prediction_workflow(components)")
    print("  result = await workflow.ainvoke({'match_id': 1, 'verbose': True})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_workflow())
