"""
Phase 3 Ensemble Evaluation

Tests ensemble prediction using multiple local LLM models:
- llama3.1:8b (primary)
- mistral:7b (alternative perspective)
- phi3:14b (larger, more capable)

Ensemble averaging typically improves accuracy by 2-4% and Brier scores by 5-10%.
"""

import asyncio
from src.evaluation.batch_evaluator import run_batch_evaluation


async def phase_3_evaluation(num_matches: int = 20):
    """
    Comprehensive evaluation of Phase 3 Ensemble improvements.

    Runs matches and compares:
    - Ensemble vs single model performance
    - Model agreement patterns
    - Confidence calibration with ensemble
    """

    print("\n" + "=" * 70)
    print("PHASE 3 ENSEMBLE EVALUATION")
    print("=" * 70)
    print(f"\nEvaluating {num_matches} matches with ensemble prediction...")
    print("Ensemble models: llama3.1:8b, mistral:7b, phi3:14b")
    print("\nExpected improvements:")
    print("  - Accuracy: +2-4%")
    print("  - Brier Score: +5-10%")
    print("  - Better calibration through model averaging")
    print("\n" + "=" * 70 + "\n")

    # Run with ensemble enabled
    evaluator, analysis = await run_batch_evaluation(
        num_matches=num_matches,
        verbose=False,
        use_ensemble=True
    )

    if evaluator is None or analysis is None:
        print("\n⚠️ Evaluation failed to complete")
        return None, None

    # Phase 3 specific analysis
    results = evaluator.results

    print("\n" + "=" * 70)
    print("PHASE 3 ENSEMBLE INSIGHTS")
    print("=" * 70)

    # Count ensemble predictions
    ensemble_preds = [r for r in results if r.get('prediction', {}).get('parse_method') == 'ensemble']
    print(f"\n📊 ENSEMBLE USAGE:")
    print(f"  Ensemble predictions: {len(ensemble_preds)}/{len(results)} ({len(ensemble_preds)/len(results)*100:.1f}%)")

    # Analyze ensemble agreement
    if ensemble_preds:
        agreements = []
        for r in ensemble_preds:
            ensemble_info = r.get('prediction', {}).get('ensemble_info', {})
            if ensemble_info and 'ensemble_confidence' in ensemble_info:
                agreements.append(ensemble_info['ensemble_confidence'])

        if agreements:
            import statistics
            avg_agreement = statistics.mean(agreements)
            print(f"\n🤝 MODEL AGREEMENT:")
            print(f"  Average agreement: {avg_agreement:.1%}")
            print(f"  Min agreement: {min(agreements):.1%}")
            print(f"  Max agreement: {max(agreements):.1%}")

            # High vs low agreement performance
            high_agree = [r for r in ensemble_preds
                         if r.get('prediction', {}).get('ensemble_info', {}).get('ensemble_confidence', 0) > 0.7]
            low_agree = [r for r in ensemble_preds
                        if r.get('prediction', {}).get('ensemble_info', {}).get('ensemble_confidence', 0) <= 0.7]

            if high_agree:
                high_correct = sum(1 for r in high_agree if r.get('llm_correct', False))
                high_acc = high_correct / len(high_agree)
                print(f"\n  High agreement (>70%): {len(high_agree)} matches, {high_acc:.1%} accuracy")

            if low_agree:
                low_correct = sum(1 for r in low_agree if r.get('llm_correct', False))
                low_acc = low_correct / len(low_agree)
                print(f"  Low agreement (≤70%): {len(low_agree)} matches, {low_acc:.1%} accuracy")

    # Compare to baseline from analysis
    baseline_acc = analysis.get('baseline_accuracy', 0)
    llm_acc = analysis.get('llm_accuracy', 0)
    improvement = llm_acc - baseline_acc

    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  Ensemble accuracy: {llm_acc:.1%}")
    print(f"  Improvement: {improvement*100:+.1f}%")

    baseline_brier = analysis.get('baseline_avg_brier', 0)
    llm_brier = analysis.get('llm_avg_brier', 0)
    brier_improvement = baseline_brier - llm_brier
    brier_pct = (brier_improvement / baseline_brier * 100) if baseline_brier > 0 else 0

    print(f"\n📉 BRIER SCORE:")
    print(f"  Baseline: {baseline_brier:.3f}")
    print(f"  Ensemble: {llm_brier:.3f}")
    print(f"  Improvement: {brier_improvement:+.3f} ({brier_pct:+.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return evaluator, analysis


if __name__ == "__main__":
    asyncio.run(phase_3_evaluation(num_matches=20))
