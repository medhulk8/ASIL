"""
Phase 3: Single Model vs Ensemble Comparison

Compare single model (llama3.1:8b) vs ensemble (3 models)
on the same matches to measure ensemble benefit.
"""

import asyncio
from src.evaluation.batch_evaluator import run_batch_evaluation


async def compare_single_vs_ensemble(num_matches: int = 30):
    """
    Compare single model (llama3.1:8b) vs ensemble (3 models)
    on the same matches.
    """

    print("\n" + "=" * 70)
    print("PHASE 3: SINGLE MODEL vs ENSEMBLE COMPARISON")
    print("=" * 70)
    print(f"\nThis will run {num_matches} matches twice:")
    print("  1. Single model (llama3.1:8b only)")
    print("  2. Ensemble (llama3.1 + mistral + phi3)")
    print("\nExpected time: ~50-60 minutes total")
    print("=" * 70 + "\n")

    # Select diverse matches (use a fixed seed for reproducibility)
    import random
    random.seed(42)
    test_matches = random.sample(range(1, 400), num_matches)

    print(f"Selected {num_matches} test matches")
    print(f"IDs: {sorted(test_matches)[:10]}... (showing first 10)")

    # Run 1: Single model
    print("\n" + "=" * 70)
    print("RUN 1: SINGLE MODEL (llama3.1:8b)")
    print("=" * 70)

    evaluator_single, analysis_single = await run_batch_evaluation(
        match_ids=test_matches,
        use_ensemble=False,  # Single model
        verbose=False
    )

    if evaluator_single is None or analysis_single is None:
        print("Single model run failed")
        return None, None, None, None

    # Run 2: Ensemble
    print("\n" + "=" * 70)
    print("RUN 2: ENSEMBLE (3 models)")
    print("=" * 70)

    evaluator_ensemble, analysis_ensemble = await run_batch_evaluation(
        match_ids=test_matches,
        use_ensemble=True,  # Ensemble
        verbose=False
    )

    if evaluator_ensemble is None or analysis_ensemble is None:
        print("Ensemble run failed")
        return evaluator_single, None, analysis_single, None

    # Comparison
    print("\n" + "=" * 70)
    print("SINGLE vs ENSEMBLE COMPARISON")
    print("=" * 70)

    print("\n📊 ACCURACY:")
    baseline_acc = analysis_single.get('baseline_accuracy', 0)
    single_acc = analysis_single.get('llm_accuracy', 0)
    ensemble_acc = analysis_ensemble.get('llm_accuracy', 0)

    print(f"  Baseline:      {baseline_acc:.1%}")
    print(f"  Single Model:  {single_acc:.1%}")
    print(f"  Ensemble:      {ensemble_acc:.1%}")

    acc_improvement = (ensemble_acc - single_acc) * 100
    print(f"  Ensemble Gain: {acc_improvement:+.1f}%")

    if acc_improvement > 2:
        print(f"  ✓ Ensemble shows significant improvement!")
    elif acc_improvement > 0:
        print(f"  ✓ Ensemble shows modest improvement")
    else:
        print(f"  ⚠ Ensemble not improving accuracy")

    print("\n📉 BRIER SCORE (lower is better):")
    baseline_brier = analysis_single.get('baseline_avg_brier', 0)
    single_brier = analysis_single.get('llm_avg_brier', 0)
    ensemble_brier = analysis_ensemble.get('llm_avg_brier', 0)

    print(f"  Baseline:      {baseline_brier:.3f}")
    print(f"  Single Model:  {single_brier:.3f}")
    print(f"  Ensemble:      {ensemble_brier:.3f}")

    brier_improvement = single_brier - ensemble_brier
    brier_improvement_pct = (brier_improvement / single_brier * 100) if single_brier > 0 else 0
    print(f"  Ensemble Gain: {brier_improvement:+.3f} ({brier_improvement_pct:+.1f}%)")

    if brier_improvement_pct > 5:
        print(f"  ✓ Excellent calibration improvement!")
    elif brier_improvement_pct > 2:
        print(f"  ✓ Good calibration improvement")
    elif brier_improvement_pct > 0:
        print(f"  ✓ Slight calibration improvement")
    else:
        print(f"  ⚠ No calibration improvement")

    print("\n🏆 HEAD-TO-HEAD (Ensemble vs Single):")

    # Match-by-match comparison
    ensemble_wins = 0
    single_wins = 0
    ties = 0

    single_results = {r['match_id']: r for r in evaluator_single.results}
    ensemble_results = {r['match_id']: r for r in evaluator_ensemble.results}

    for match_id in test_matches:
        single_brier_match = single_results.get(match_id, {}).get('llm_brier', 999)
        ensemble_brier_match = ensemble_results.get(match_id, {}).get('llm_brier', 999)

        if abs(ensemble_brier_match - single_brier_match) < 0.01:
            ties += 1
        elif ensemble_brier_match < single_brier_match:
            ensemble_wins += 1
        else:
            single_wins += 1

    total_decided = ensemble_wins + single_wins
    ensemble_win_rate = ensemble_wins / total_decided if total_decided > 0 else 0

    print(f"  Ensemble wins: {ensemble_wins} matches")
    print(f"  Single wins:   {single_wins} matches")
    print(f"  Ties:          {ties} matches")
    print(f"  Ensemble win rate: {ensemble_win_rate:.1%}")

    if ensemble_win_rate > 0.55:
        print(f"  ✓ Ensemble consistently better!")
    elif ensemble_win_rate > 0.50:
        print(f"  ✓ Ensemble slightly better")
    else:
        print(f"  ≈ No clear winner")

    print("\n⏱️ EFFICIENCY ANALYSIS:")
    print(f"  Single model:  ~20-25s per match")
    print(f"  Ensemble:      ~50-60s per match (3x predictions)")
    time_multiplier = 2.5
    print(f"  Time cost: {time_multiplier:.1f}x slower")
    print(f"  Accuracy gain: {acc_improvement:+.1f}%")
    efficiency_ratio = acc_improvement / time_multiplier if time_multiplier > 0 else 0
    print(f"  Efficiency ratio: {efficiency_ratio:.2f}% accuracy per 1x time")

    # Confidence distribution comparison
    print("\n🎲 CONFIDENCE DISTRIBUTION:")
    single_conf = {}
    ensemble_conf = {}

    for r in evaluator_single.results:
        conf = r.get('llm_confidence', 'unknown')
        single_conf[conf] = single_conf.get(conf, 0) + 1

    for r in evaluator_ensemble.results:
        conf = r.get('llm_confidence', 'unknown')
        ensemble_conf[conf] = ensemble_conf.get(conf, 0) + 1

    print("  Single Model:")
    for conf in ['high', 'medium', 'low']:
        count = single_conf.get(conf, 0)
        pct = count / len(evaluator_single.results) * 100 if evaluator_single.results else 0
        print(f"    {conf}: {count} ({pct:.0f}%)")

    print("  Ensemble:")
    for conf in ['high', 'medium', 'low']:
        count = ensemble_conf.get(conf, 0)
        pct = count / len(evaluator_ensemble.results) * 100 if evaluator_ensemble.results else 0
        print(f"    {conf}: {count} ({pct:.0f}%)")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if acc_improvement > 2 and brier_improvement_pct > 5:
        print("\n✓ ENSEMBLE is significantly better")
        print("  Recommend: Use ensemble for final production system")
        recommendation = "ENSEMBLE"
    elif acc_improvement > 1 or brier_improvement_pct > 3:
        print("\n✓ ENSEMBLE shows improvement")
        print("  Recommend: Use ensemble when time permits")
        recommendation = "ENSEMBLE (when time permits)"
    elif acc_improvement > 0 or brier_improvement_pct > 0:
        print("\n≈ ENSEMBLE shows marginal benefit")
        print("  Recommend: Consider ensemble for important predictions")
        recommendation = "OPTIONAL"
    else:
        print("\n≈ ENSEMBLE shows no clear benefit")
        print("  Recommend: Single model sufficient for most use cases")
        recommendation = "SINGLE MODEL"

    print("\n" + "=" * 70)
    print("PHASE 3 COMPARISON COMPLETE")
    print("=" * 70)

    # Summary dict
    summary = {
        'baseline_accuracy': baseline_acc,
        'single_accuracy': single_acc,
        'ensemble_accuracy': ensemble_acc,
        'accuracy_improvement': acc_improvement,
        'baseline_brier': baseline_brier,
        'single_brier': single_brier,
        'ensemble_brier': ensemble_brier,
        'brier_improvement': brier_improvement,
        'brier_improvement_pct': brier_improvement_pct,
        'ensemble_wins': ensemble_wins,
        'single_wins': single_wins,
        'ensemble_win_rate': ensemble_win_rate,
        'recommendation': recommendation
    }

    return evaluator_single, evaluator_ensemble, analysis_single, analysis_ensemble, summary


if __name__ == "__main__":
    asyncio.run(compare_single_vs_ensemble(num_matches=30))
