"""
Phase 5: Granular Analysis

Analyzes existing evaluation results to identify patterns and insights:
- Accuracy by category (home wins, draws, away wins)
- Accuracy by outcome type
- Systematic failure patterns
- Confidence calibration analysis
- Insights report generation

No new API calls - analyzes existing CSV data only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class GranularAnalyzer:
    """Analyzes evaluation results in detail"""

    def __init__(self, csv_path: str = "data/evaluation_results.csv"):
        self.csv_path = csv_path
        self.df = None
        self.insights = {}

    def load_data(self) -> bool:
        """Load evaluation results from CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded {len(self.df)} evaluation results from {self.csv_path}")
            return True
        except FileNotFoundError:
            print(f"✗ File not found: {self.csv_path}")
            return False
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return False

    def analyze_by_outcome_category(self) -> Dict:
        """Calculate accuracy broken down by actual outcome type (H/D/A)"""
        print("\n" + "=" * 70)
        print("ACCURACY BY OUTCOME CATEGORY")
        print("=" * 70)

        results = {}

        for outcome in ['H', 'D', 'A']:
            outcome_df = self.df[self.df['actual_outcome'] == outcome]

            if len(outcome_df) == 0:
                continue

            baseline_correct = outcome_df['baseline_correct'].sum()
            llm_correct = outcome_df['llm_correct'].sum()
            total = len(outcome_df)

            baseline_acc = baseline_correct / total if total > 0 else 0
            llm_acc = llm_correct / total if total > 0 else 0
            improvement = llm_acc - baseline_acc

            outcome_name = {'H': 'Home Wins', 'D': 'Draws', 'A': 'Away Wins'}[outcome]

            print(f"\n{outcome_name} ({outcome}):")
            print(f"  Matches: {total}")
            print(f"  Baseline accuracy: {baseline_acc:.1%} ({baseline_correct}/{total})")
            print(f"  LLM accuracy: {llm_acc:.1%} ({llm_correct}/{total})")
            print(f"  Improvement: {improvement*100:+.1f}%")

            results[outcome] = {
                'total': total,
                'baseline_accuracy': baseline_acc,
                'llm_accuracy': llm_acc,
                'improvement': improvement,
                'baseline_correct': int(baseline_correct),
                'llm_correct': int(llm_correct)
            }

        self.insights['by_outcome'] = results
        return results

    def analyze_by_confidence_level(self) -> Dict:
        """Analyze accuracy by LLM confidence level"""
        print("\n" + "=" * 70)
        print("ACCURACY BY CONFIDENCE LEVEL")
        print("=" * 70)

        results = {}

        for conf_level in ['low', 'medium', 'high']:
            conf_df = self.df[self.df['llm_confidence'] == conf_level]

            if len(conf_df) == 0:
                print(f"\n{conf_level.upper()}: No predictions")
                continue

            llm_correct = conf_df['llm_correct'].sum()
            total = len(conf_df)
            accuracy = llm_correct / total if total > 0 else 0

            # Average Brier score
            avg_brier = conf_df['llm_brier'].mean()

            print(f"\n{conf_level.upper()}:")
            print(f"  Matches: {total} ({total/len(self.df)*100:.1f}%)")
            print(f"  Accuracy: {accuracy:.1%} ({llm_correct}/{total})")
            print(f"  Avg Brier Score: {avg_brier:.3f}")

            results[conf_level] = {
                'total': total,
                'accuracy': accuracy,
                'correct': int(llm_correct),
                'avg_brier': float(avg_brier),
                'percentage': total / len(self.df)
            }

        self.insights['by_confidence'] = results
        return results

    def analyze_web_search_impact(self) -> Dict:
        """Analyze impact of web searches on accuracy"""
        print("\n" + "=" * 70)
        print("WEB SEARCH IMPACT")
        print("=" * 70)

        # With web searches
        with_search = self.df[self.df['skipped_web_search'] == False]
        without_search = self.df[self.df['skipped_web_search'] == True]

        results = {}

        if len(with_search) > 0:
            search_acc = with_search['llm_correct'].sum() / len(with_search)
            search_brier = with_search['llm_brier'].mean()
            avg_searches = with_search['web_searches_performed'].mean()

            print(f"\nWITH Web Search ({len(with_search)} matches):")
            print(f"  Accuracy: {search_acc:.1%}")
            print(f"  Avg Brier Score: {search_brier:.3f}")
            print(f"  Avg searches per match: {avg_searches:.1f}")

            results['with_search'] = {
                'total': len(with_search),
                'accuracy': search_acc,
                'avg_brier': float(search_brier),
                'avg_searches': float(avg_searches)
            }

        if len(without_search) > 0:
            no_search_acc = without_search['llm_correct'].sum() / len(without_search)
            no_search_brier = without_search['llm_brier'].mean()

            print(f"\nWITHOUT Web Search ({len(without_search)} matches):")
            print(f"  Accuracy: {no_search_acc:.1%}")
            print(f"  Avg Brier Score: {no_search_brier:.3f}")

            results['without_search'] = {
                'total': len(without_search),
                'accuracy': no_search_acc,
                'avg_brier': float(no_search_brier)
            }

        if len(with_search) > 0 and len(without_search) > 0:
            acc_impact = search_acc - no_search_acc
            brier_impact = no_search_brier - search_brier  # Lower is better

            print(f"\nIMPACT:")
            print(f"  Accuracy improvement: {acc_impact*100:+.1f}%")
            print(f"  Brier score improvement: {brier_impact:+.3f}")

            results['impact'] = {
                'accuracy_improvement': acc_impact,
                'brier_improvement': brier_impact
            }

        self.insights['web_search'] = results
        return results

    def identify_failure_patterns(self) -> Dict:
        """Identify systematic patterns in LLM failures"""
        print("\n" + "=" * 70)
        print("FAILURE PATTERN ANALYSIS")
        print("=" * 70)

        # Get failures
        failures = self.df[self.df['llm_correct'] == False]

        if len(failures) == 0:
            print("\n✓ No failures to analyze!")
            return {}

        print(f"\nTotal failures: {len(failures)}/{len(self.df)} ({len(failures)/len(self.df)*100:.1f}%)")

        results = {}

        # 1. Failures by outcome type
        print("\nFailures by actual outcome:")
        for outcome in ['H', 'D', 'A']:
            outcome_failures = failures[failures['actual_outcome'] == outcome]
            outcome_total = len(self.df[self.df['actual_outcome'] == outcome])

            if outcome_total > 0:
                fail_rate = len(outcome_failures) / outcome_total
                outcome_name = {'H': 'Home Wins', 'D': 'Draws', 'A': 'Away Wins'}[outcome]
                print(f"  {outcome_name}: {len(outcome_failures)}/{outcome_total} ({fail_rate:.1%})")

        # 2. Failures by confidence level
        print("\nFailures by confidence level:")
        for conf in ['low', 'medium', 'high']:
            conf_failures = failures[failures['llm_confidence'] == conf]
            conf_total = len(self.df[self.df['llm_confidence'] == conf])

            if conf_total > 0:
                fail_rate = len(conf_failures) / conf_total
                print(f"  {conf.upper()}: {len(conf_failures)}/{conf_total} ({fail_rate:.1%})")

        # 3. Overconfident failures (high confidence but wrong)
        overconfident = failures[failures['llm_confidence'] == 'high']
        if len(overconfident) > 0:
            print(f"\n⚠️  OVERCONFIDENT FAILURES: {len(overconfident)}")
            print("  High confidence predictions that were wrong")

            # Show examples
            for idx, row in overconfident.head(3).iterrows():
                print(f"\n  Example: {row['home_team']} vs {row['away_team']}")
                print(f"    Predicted: {self._get_prediction(row)} | Actual: {row['actual_outcome']}")
                print(f"    LLM probs: H:{row['llm_home']:.2f} D:{row['llm_draw']:.2f} A:{row['llm_away']:.2f}")

        # 4. Brier score analysis of failures
        avg_failure_brier = failures['llm_brier'].mean()
        avg_success_brier = self.df[self.df['llm_correct'] == True]['llm_brier'].mean()

        print(f"\nBrier scores:")
        print(f"  Failed predictions: {avg_failure_brier:.3f}")
        print(f"  Successful predictions: {avg_success_brier:.3f}")
        print(f"  Difference: {avg_failure_brier - avg_success_brier:+.3f}")

        results['total_failures'] = len(failures)
        results['failure_rate'] = len(failures) / len(self.df)
        results['overconfident_failures'] = len(overconfident)
        results['avg_failure_brier'] = float(avg_failure_brier)
        results['avg_success_brier'] = float(avg_success_brier)

        self.insights['failures'] = results
        return results

    def _get_prediction(self, row) -> str:
        """Get the predicted outcome from probabilities"""
        probs = {
            'H': row['llm_home'],
            'D': row['llm_draw'],
            'A': row['llm_away']
        }
        return max(probs, key=probs.get)

    def analyze_baseline_vs_llm(self) -> Dict:
        """Compare where LLM and baseline agree/disagree"""
        print("\n" + "=" * 70)
        print("BASELINE vs LLM COMPARISON")
        print("=" * 70)

        # Both correct
        both_correct = self.df[(self.df['baseline_correct'] == True) & (self.df['llm_correct'] == True)]
        # Only baseline correct
        only_baseline = self.df[(self.df['baseline_correct'] == True) & (self.df['llm_correct'] == False)]
        # Only LLM correct
        only_llm = self.df[(self.df['baseline_correct'] == False) & (self.df['llm_correct'] == True)]
        # Both wrong
        both_wrong = self.df[(self.df['baseline_correct'] == False) & (self.df['llm_correct'] == False)]

        total = len(self.df)

        print(f"\nAgreement patterns:")
        print(f"  Both correct: {len(both_correct)}/{total} ({len(both_correct)/total*100:.1f}%)")
        print(f"  Only baseline correct: {len(only_baseline)}/{total} ({len(only_baseline)/total*100:.1f}%)")
        print(f"  Only LLM correct: {len(only_llm)}/{total} ({len(only_llm)/total*100:.1f}%)")
        print(f"  Both wrong: {len(both_wrong)}/{total} ({len(both_wrong)/total*100:.1f}%)")

        # Show where LLM uniquely succeeded
        if len(only_llm) > 0:
            print(f"\n✓ LLM unique wins ({len(only_llm)} matches):")
            for idx, row in only_llm.head(3).iterrows():
                print(f"  • {row['home_team']} vs {row['away_team']} → {row['actual_outcome']}")
                print(f"    Baseline: H:{row['baseline_home']:.2f} D:{row['baseline_draw']:.2f} A:{row['baseline_away']:.2f}")
                print(f"    LLM: H:{row['llm_home']:.2f} D:{row['llm_draw']:.2f} A:{row['llm_away']:.2f}")

        results = {
            'both_correct': len(both_correct),
            'only_baseline': len(only_baseline),
            'only_llm': len(only_llm),
            'both_wrong': len(both_wrong),
            'total': total
        }

        self.insights['baseline_vs_llm'] = results
        return results

    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report"""
        print("\n" + "=" * 70)
        print("INSIGHTS SUMMARY")
        print("=" * 70)

        report = []

        # Overall performance
        baseline_acc = self.df['baseline_correct'].sum() / len(self.df)
        llm_acc = self.df['llm_correct'].sum() / len(self.df)
        improvement = llm_acc - baseline_acc

        report.append(f"\n📊 OVERALL PERFORMANCE")
        report.append(f"  Dataset: {len(self.df)} matches")
        report.append(f"  Baseline accuracy: {baseline_acc:.1%}")
        report.append(f"  LLM accuracy: {llm_acc:.1%}")
        report.append(f"  Improvement: {improvement*100:+.1f}%")

        # Key insights
        report.append(f"\n🔍 KEY INSIGHTS")

        # 1. Outcome-specific performance
        if 'by_outcome' in self.insights:
            best_outcome = max(self.insights['by_outcome'].items(),
                             key=lambda x: x[1]['llm_accuracy'])
            worst_outcome = min(self.insights['by_outcome'].items(),
                              key=lambda x: x[1]['llm_accuracy'])

            outcome_names = {'H': 'home wins', 'D': 'draws', 'A': 'away wins'}
            report.append(f"  1. Best at predicting: {outcome_names[best_outcome[0]]} ({best_outcome[1]['llm_accuracy']:.1%})")
            report.append(f"     Worst at predicting: {outcome_names[worst_outcome[0]]} ({worst_outcome[1]['llm_accuracy']:.1%})")

        # 2. Confidence calibration
        if 'by_confidence' in self.insights:
            conf_insights = self.insights['by_confidence']

            if 'high' in conf_insights:
                high_acc = conf_insights['high']['accuracy']
                report.append(f"  2. High confidence predictions: {high_acc:.1%} accuracy ({conf_insights['high']['total']} matches)")

            if 'low' in conf_insights:
                low_acc = conf_insights['low']['accuracy']
                report.append(f"     Low confidence predictions: {low_acc:.1%} accuracy ({conf_insights['low']['total']} matches)")

        # 3. Web search impact
        if 'web_search' in self.insights and 'impact' in self.insights['web_search']:
            impact = self.insights['web_search']['impact']
            report.append(f"  3. Web search impact: {impact['accuracy_improvement']*100:+.1f}% accuracy")
            report.append(f"     Brier improvement: {impact['brier_improvement']:+.3f}")

        # 4. Failure patterns
        if 'failures' in self.insights:
            fail_info = self.insights['failures']
            if fail_info.get('overconfident_failures', 0) > 0:
                report.append(f"  4. ⚠️ Overconfident failures: {fail_info['overconfident_failures']}")
                report.append(f"     (High confidence but wrong predictions)")

        # 5. LLM unique wins
        if 'baseline_vs_llm' in self.insights:
            comp = self.insights['baseline_vs_llm']
            report.append(f"  5. LLM unique wins: {comp['only_llm']} (baseline got wrong, LLM got right)")
            report.append(f"     LLM unique losses: {comp['only_baseline']} (baseline got right, LLM got wrong)")

        report_text = "\n".join(report)
        print(report_text)

        # Save to file
        report_path = "data/phase_5_insights.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PHASE 5: GRANULAR ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(report_text)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("DETAILED INSIGHTS (JSON)\n")
            f.write("=" * 70 + "\n")
            f.write(json.dumps(self.insights, indent=2))

        print(f"\n✓ Insights report saved to: {report_path}")

        return report_text

    def run_full_analysis(self):
        """Run complete granular analysis"""
        if not self.load_data():
            return False

        print(f"\n{'='*70}")
        print(f"PHASE 5: GRANULAR ANALYSIS")
        print(f"Analyzing {len(self.df)} predictions")
        print(f"{'='*70}")

        # Run all analyses
        self.analyze_by_outcome_category()
        self.analyze_by_confidence_level()
        self.analyze_web_search_impact()
        self.analyze_baseline_vs_llm()
        self.identify_failure_patterns()
        self.generate_insights_report()

        print(f"\n{'='*70}")
        print("PHASE 5 COMPLETE")
        print(f"{'='*70}\n")

        return True


if __name__ == "__main__":
    analyzer = GranularAnalyzer()
    analyzer.run_full_analysis()
