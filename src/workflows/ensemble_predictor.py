"""
Ensemble Predictor for ASIL Predictions

Combines predictions from multiple LLM models for improved accuracy.
Ensemble averaging typically improves accuracy by 2-4% and Brier scores by 5-10%.

Models used:
- llama3.1:8b (primary, balanced)
- mistral:7b (alternative perspective)
- phi3:14b (larger, more capable)
"""

import ollama
from typing import Dict, List, Optional
import statistics
import re


class EnsemblePredictor:
    """
    Ensemble prediction using multiple LLM models.

    Combines predictions from multiple models and averages probabilities
    for more robust predictions.
    """

    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize ensemble predictor.

        Args:
            models: List of model names (default: llama3.1:8b, mistral:7b, phi3:14b)
        """
        self.models = models or [
            "llama3.1:8b",
            "mistral:7b",
            "phi3:14b"
        ]

        # Verify models are available
        self._verify_models()

    def _verify_models(self):
        """Check all models are available in Ollama."""
        try:
            available = ollama.list()
            # Handle both old dict format and new ListResponse format
            if hasattr(available, 'models'):
                available_names = [m.model for m in available.models]
            else:
                available_names = [m.get('name', m.get('model', '')) for m in available.get('models', [])]

            missing = []
            for model in self.models:
                # Check if model or model base name is available
                found = False
                for avail in available_names:
                    if model == avail or model in avail or avail.startswith(model.split(':')[0]):
                        found = True
                        break
                if not found:
                    missing.append(model)

            if missing:
                print(f"Warning: Models not found: {missing}")
                print("Run: ollama pull <model_name> for each missing model")
                # Remove missing models
                self.models = [m for m in self.models if m not in missing]

            if self.models:
                print(f"Ensemble: {len(self.models)} models verified")
            else:
                raise ValueError("No models available for ensemble")

        except Exception as e:
            print(f"Model verification error: {e}")
            raise

    def predict_ensemble(
        self,
        context: str,
        verbose: bool = False
    ) -> Dict:
        """
        Get ensemble prediction from multiple models.

        Args:
            context: Full LLM prompt context
            verbose: Print individual model predictions

        Returns:
            dict with:
            - home_prob, draw_prob, away_prob (averaged)
            - individual_predictions (list of each model's prediction)
            - reasoning (combined reasoning)
            - ensemble_confidence (agreement score)
        """
        individual_predictions = []

        if verbose:
            print(f"\n  Running ensemble prediction ({len(self.models)} models)...")

        # Get prediction from each model
        for i, model in enumerate(self.models, 1):
            if verbose:
                print(f"    [{i}/{len(self.models)}] {model}...", end=" ", flush=True)

            try:
                prediction = self._get_single_prediction(model, context)
                individual_predictions.append({
                    'model': model,
                    'home_prob': prediction['home_prob'],
                    'draw_prob': prediction['draw_prob'],
                    'away_prob': prediction['away_prob'],
                    'reasoning': prediction['reasoning']
                })

                if verbose:
                    print(f"H={prediction['home_prob']:.1%} D={prediction['draw_prob']:.1%} A={prediction['away_prob']:.1%}")

            except Exception as e:
                if verbose:
                    print(f"Failed: {e}")
                # Skip failed model, continue with others
                continue

        if not individual_predictions:
            raise ValueError("All models failed to generate predictions")

        # Average probabilities
        avg_home = statistics.mean([p['home_prob'] for p in individual_predictions])
        avg_draw = statistics.mean([p['draw_prob'] for p in individual_predictions])
        avg_away = statistics.mean([p['away_prob'] for p in individual_predictions])

        # Normalize to ensure sum = 1.0
        total = avg_home + avg_draw + avg_away
        avg_home /= total
        avg_draw /= total
        avg_away /= total

        # Calculate ensemble confidence (agreement between models)
        confidence = self._calculate_ensemble_confidence(individual_predictions)

        # Combine reasoning
        combined_reasoning = self._combine_reasoning(individual_predictions)

        result = {
            'home_prob': avg_home,
            'draw_prob': avg_draw,
            'away_prob': avg_away,
            'individual_predictions': individual_predictions,
            'reasoning': combined_reasoning,
            'ensemble_confidence': confidence,
            'num_models': len(individual_predictions)
        }

        if verbose:
            print(f"    Ensemble: H={avg_home:.1%} D={avg_draw:.1%} A={avg_away:.1%}")
            print(f"    Agreement: {confidence:.1%}")

        return result

    def _get_single_prediction(self, model: str, context: str) -> Dict:
        """
        Get prediction from a single model.
        """
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": context}]
        )

        response_text = response['message']['content']

        # Parse prediction using same logic as workflow
        prediction = self._parse_prediction(response_text)

        return prediction

    def _parse_prediction(self, response_text: str) -> Dict:
        """
        Parse LLM response to extract probabilities.

        Handles multiple formats:
        - PROBABILITIES: H=X%, D=Y%, A=Z%
        - Home: X%, Draw: Y%, Away: Z%
        - Natural language with percentages
        """
        # Try format 1: H=X%, D=Y%, A=Z%
        pattern1 = r'H\s*=\s*(\d+(?:\.\d+)?)\s*%.*?D\s*=\s*(\d+(?:\.\d+)?)\s*%.*?A\s*=\s*(\d+(?:\.\d+)?)\s*%'
        match = re.search(pattern1, response_text, re.IGNORECASE | re.DOTALL)

        if match:
            home = float(match.group(1)) / 100
            draw = float(match.group(2)) / 100
            away = float(match.group(3)) / 100
        else:
            # Try format 2: Home: X%, Draw: Y%, Away: Z%
            pattern2 = r'Home.*?(\d+(?:\.\d+)?)\s*%.*?Draw.*?(\d+(?:\.\d+)?)\s*%.*?Away.*?(\d+(?:\.\d+)?)\s*%'
            match = re.search(pattern2, response_text, re.IGNORECASE | re.DOTALL)

            if match:
                home = float(match.group(1)) / 100
                draw = float(match.group(2)) / 100
                away = float(match.group(3)) / 100
            else:
                # Fallback: find any 3 numbers that look like probabilities
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', response_text)
                if len(numbers) >= 3:
                    home = float(numbers[0]) / 100
                    draw = float(numbers[1]) / 100
                    away = float(numbers[2]) / 100
                else:
                    # Ultimate fallback: equal probabilities
                    home, draw, away = 0.33, 0.34, 0.33

        # Normalize
        total = home + draw + away
        if total > 0:
            home /= total
            draw /= total
            away /= total
        else:
            home, draw, away = 0.33, 0.34, 0.33

        # Extract reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?:CONFIDENCE:|$)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        return {
            'home_prob': home,
            'draw_prob': draw,
            'away_prob': away,
            'reasoning': reasoning[:200]  # Limit length
        }

    def _calculate_ensemble_confidence(self, predictions: List[Dict]) -> float:
        """
        Calculate how much models agree with each other.

        High agreement = high confidence
        Low agreement = low confidence

        Returns: 0.0-1.0 score
        """
        if len(predictions) < 2:
            return 1.0

        # Calculate variance in predictions
        home_probs = [p['home_prob'] for p in predictions]
        draw_probs = [p['draw_prob'] for p in predictions]
        away_probs = [p['away_prob'] for p in predictions]

        # Lower variance = higher agreement
        home_var = statistics.variance(home_probs) if len(home_probs) > 1 else 0
        draw_var = statistics.variance(draw_probs) if len(draw_probs) > 1 else 0
        away_var = statistics.variance(away_probs) if len(away_probs) > 1 else 0

        avg_variance = (home_var + draw_var + away_var) / 3

        # Convert variance to agreement score (0-1)
        # Low variance (< 0.01) = high agreement (> 0.9)
        # High variance (> 0.05) = low agreement (< 0.5)
        if avg_variance < 0.005:
            agreement = 0.95
        elif avg_variance < 0.01:
            agreement = 0.85
        elif avg_variance < 0.02:
            agreement = 0.70
        elif avg_variance < 0.03:
            agreement = 0.55
        elif avg_variance < 0.05:
            agreement = 0.40
        else:
            agreement = 0.25

        return agreement

    def _combine_reasoning(self, predictions: List[Dict]) -> str:
        """
        Combine reasoning from multiple models.
        """
        if not predictions:
            return "No reasoning available"

        # Use the first model's reasoning (usually llama3.1)
        # Could be enhanced to synthesize multiple reasonings
        return predictions[0]['reasoning']

    def get_ensemble_stats(self, predictions: List[Dict]) -> Dict:
        """
        Get statistics about the ensemble predictions.

        Useful for analysis and debugging.
        """
        if not predictions:
            return {}

        home_probs = [p['home_prob'] for p in predictions]
        draw_probs = [p['draw_prob'] for p in predictions]
        away_probs = [p['away_prob'] for p in predictions]

        return {
            'home': {
                'mean': statistics.mean(home_probs),
                'stdev': statistics.stdev(home_probs) if len(home_probs) > 1 else 0,
                'min': min(home_probs),
                'max': max(home_probs),
                'range': max(home_probs) - min(home_probs)
            },
            'draw': {
                'mean': statistics.mean(draw_probs),
                'stdev': statistics.stdev(draw_probs) if len(draw_probs) > 1 else 0,
                'min': min(draw_probs),
                'max': max(draw_probs),
                'range': max(draw_probs) - min(draw_probs)
            },
            'away': {
                'mean': statistics.mean(away_probs),
                'stdev': statistics.stdev(away_probs) if len(away_probs) > 1 else 0,
                'min': min(away_probs),
                'max': max(away_probs),
                'range': max(away_probs) - min(away_probs)
            },
            'num_models': len(predictions)
        }


def test_ensemble():
    """Test ensemble prediction on sample context."""

    print("\n" + "=" * 70)
    print("ENSEMBLE PREDICTOR TEST")
    print("=" * 70)

    # Simple test context
    test_context = """You are an expert football analyst. Predict the match outcome.

MATCH: Manchester City vs Burnley
DATE: 2024-01-15
VENUE: Manchester City (Home)

RECENT FORM:
- Manchester City: 5W-0D-0L (excellent form, 15 points from last 5)
- Burnley: 0W-1D-4L (terrible form, 1 point from last 5)

BASELINE PROBABILITIES:
- Home win: 92%
- Draw: 5%
- Away win: 3%

Based on this analysis, provide your prediction.

Respond EXACTLY in this format:
PROBABILITIES: H=XX%, D=XX%, A=XX%
REASONING: [2-3 sentences explaining your prediction]
"""

    try:
        ensemble = EnsemblePredictor()

        print("\nRunning ensemble prediction...")
        result = ensemble.predict_ensemble(test_context, verbose=True)

        print("\n" + "=" * 70)
        print("ENSEMBLE RESULT:")
        print("=" * 70)
        print(f"Final Probabilities:")
        print(f"  Home: {result['home_prob']:.1%}")
        print(f"  Draw: {result['draw_prob']:.1%}")
        print(f"  Away: {result['away_prob']:.1%}")
        print(f"\nEnsemble Agreement: {result['ensemble_confidence']:.1%}")
        print(f"Models Used: {result['num_models']}")

        # Show individual predictions
        print("\n" + "-" * 50)
        print("Individual Model Predictions:")
        for pred in result['individual_predictions']:
            print(f"  {pred['model']}: H={pred['home_prob']:.1%} D={pred['draw_prob']:.1%} A={pred['away_prob']:.1%}")

        # Get stats
        stats = ensemble.get_ensemble_stats(result['individual_predictions'])
        print("\n" + "-" * 50)
        print("Prediction Variance:")
        print(f"  Home range: {stats['home']['min']:.1%} - {stats['home']['max']:.1%} (spread: {stats['home']['range']:.1%})")
        print(f"  Draw range: {stats['draw']['min']:.1%} - {stats['draw']['max']:.1%} (spread: {stats['draw']['range']:.1%})")
        print(f"  Away range: {stats['away']['min']:.1%} - {stats['away']['max']:.1%} (spread: {stats['away']['range']:.1%})")

        print("\n" + "=" * 70)
        print("ENSEMBLE TEST COMPLETE")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_ensemble()
