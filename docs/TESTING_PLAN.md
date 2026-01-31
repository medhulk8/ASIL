# Large-Scale Testing Plan

**Objective:** Validate Phase 3/5/6 improvements on a statistically significant dataset

---

## Test Configuration

### Test 1: Current (30 matches) ⏳
- **Status:** Running (Task: b8b34bf)
- **Matches:** 30
- **Strategy:** Minimal search (default)
- **Purpose:** Quick validation after default change

### Test 2: Large-Scale (100 matches) 📋
- **Status:** Ready to run
- **Matches:** 100
- **Strategy:** Minimal search
- **Purpose:** Comprehensive statistical validation
- **Command:**
  ```bash
  export TAVILY_API_KEY="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" && \
  python run_large_test.py
  ```

---

## Success Criteria

### Critical Metrics (Must Pass)

| Metric | Target | Baseline |
|--------|--------|----------|
| **Overall Accuracy** | ≥65% | Was 48-50% |
| **Draw Accuracy** | ≥40% | Was 12.5% |
| **Web Search Impact** | 0% ± 5% | Was -18.5% |
| **Searches/Match** | 1-2 | Was 5.0 |

### Performance Metrics (Goals)

| Metric | Target | Baseline |
|--------|--------|----------|
| **Time/Match** | ≤25s | Was 32.6s |
| **Low Conf Accuracy** | ≥80% | - |
| **Brier Score** | ≤0.55 | Was 0.518 |

---

## What We're Validating

### 1. Minimal Search Strategy ✅
**Change:** Default changed to `strategy="minimal"`
- Expect: 1-2 searches per match (vs 5.0)
- Expect: Faster execution (~40%)
- Expect: Maintained or improved accuracy

### 2. Enhanced Draw Detection ✅
**Changes:**
- More aggressive thresholds (0.3→0.35 max)
- Directive warnings
- Minimum draw enforcement

**Expected:**
- Draw accuracy: 40-50% (vs 12.5%)
- Better identification of close matches
- Draw probabilities 30-40% (vs 20-25%)

### 3. Overall System Integration
**Combined effect of all improvements:**
- Web search now neutral/helpful (vs -18.5% hurt)
- Better calibration (low conf = high accuracy)
- Improved draw detection compensating for any losses

---

## Expected Results (100 matches)

### Optimistic Scenario
```
Overall Accuracy:     70-75%
Draw Accuracy:        45-55%
Home Win Accuracy:    75-80%
Away Win Accuracy:    65-70%
Searches/Match:       1.2-1.5
Time/Match:           18-22s
Web Search Impact:    +2-5%
```

### Realistic Scenario
```
Overall Accuracy:     65-70%
Draw Accuracy:        40-45%
Home Win Accuracy:    70-75%
Away Win Accuracy:    60-65%
Searches/Match:       1.5-2.0
Time/Match:           22-26s
Web Search Impact:    0 ± 3%
```

### Minimum Acceptable
```
Overall Accuracy:     ≥65%
Draw Accuracy:        ≥40%
Home Win Accuracy:    ≥65%
Away Win Accuracy:    ≥60%
Searches/Match:       ≤2.5
Time/Match:           ≤28s
Web Search Impact:    ≥ -5%
```

---

## Analysis Plan

### Immediate Analysis (After Test Completes)
1. Run Phase 5 granular analysis
   ```bash
   python -m src.evaluation.phase_5_granular_analysis
   ```

2. Check key metrics:
   - Overall accuracy (target: ≥65%)
   - Draw accuracy (target: ≥40%)
   - Searches per match (target: 1-2)
   - Web search impact (target: neutral)

3. Compare to baseline:
   - Phase 3 results (48-50% accuracy)
   - Phase 5 results (web search -18.5%)
   - First validation (66.7% accuracy, 50% draws)

### Detailed Analysis
4. **Outcome-specific performance:**
   - Home wins, draws, away wins breakdown
   - Identify which improved, which regressed

5. **Confidence calibration:**
   - Accuracy by confidence level
   - Verify low confidence = high accuracy

6. **Search efficiency:**
   - Confirm minimal queries (1-2 not 5)
   - Check if searches help or hurt
   - Breakdown by search vs no-search

7. **Failure patterns:**
   - What types of matches still fail?
   - Are there systematic blindspots?
   - Overconfident errors?

---

## Potential Issues & Mitigations

### Issue 1: Draw Accuracy Regresses
**If:** Draw accuracy < 35%
**Action:**
- Check if draw warnings are appearing
- Verify draw_likelihood scores
- May need even more aggressive thresholds

### Issue 2: Search Count Still High
**If:** Searches/match > 3.0
**Action:**
- Verify default changed to "minimal"
- Check workflow isn't overriding
- May need to explicitly pass strategy param

### Issue 3: Overall Accuracy Drops
**If:** Overall accuracy < 60%
**Action:**
- Analyze what changed vs validation
- Check if certain match types regressed
- May be dataset variance (30→100 matches)

### Issue 4: Time Regression
**If:** Time/match > 30s
**Action:**
- Check if search count is high
- Verify minimal strategy active
- May need to optimize other components

---

## Post-Test Actions

### If Results Are Good (Targets Met) ✅
1. **Document:** Update all docs with 100-match results
2. **Commit:** Commit all improvements with results
3. **Deploy:** Mark system as production-ready
4. **Next:** Consider Phase 7+ improvements (if any)

### If Results Need Tuning ⚙️
1. **Analyze:** Identify specific regressions
2. **Adjust:** Fine-tune thresholds/strategies
3. **Re-test:** Run targeted 20-30 match tests
4. **Iterate:** Repeat until targets met

### If Major Issues Found ❌
1. **Investigate:** Deep dive into failure modes
2. **Diagnose:** Determine root cause
3. **Fix:** Apply targeted fixes
4. **Validate:** Re-run full test

---

## Statistical Significance

### 30-Match Test
- Confidence: ~85% (small sample)
- Use for: Quick validation, trend checking
- Limited by: High variance

### 100-Match Test
- Confidence: ~95% (large sample)
- Use for: Production decisions, final validation
- Statistical power: Good for ±5% accuracy differences

**Why 100 matches:**
- Sufficient for statistical significance
- ~25 draws expected (good draw sample)
- Covers early/mid/late season
- Balanced home/away/draw distribution
- Reasonable test time (~30-60 minutes)

---

## Monitoring Commands

### Check Progress
```bash
# Current 30-match test
tail -f /private/tmp/claude-502/-Users-medhul-asil-project/tasks/b8b34bf.output

# Future 100-match test
tail -f /private/tmp/claude-502/-Users-medhul-asil-project/tasks/[TASK_ID].output
```

### Quick Status
```bash
# See last 50 lines
tail -n 50 /private/tmp/claude-502/-Users-medhul-asil-project/tasks/b8b34bf.output

# Filter for key info
tail -n 200 [output_file] | grep -E "(Processing|accuracy|Brier|searches)"
```

### After Completion
```bash
# Run analysis
python -m src.evaluation.phase_5_granular_analysis

# View results
cat data/evaluation_results.csv | tail -n 110 | head -n 100

# Check insights
cat data/phase_5_insights.txt
```

---

## Timeline

### Current Status (17:45)
- ⏳ 30-match test running (started 17:45)
- 📋 100-match test ready to launch
- ⏱️ Estimated completion: ~18:15 (30-match), ~19:30 (100-match)

### Next Steps
1. Wait for 30-match completion (~30 min)
2. Analyze 30-match results
3. Launch 100-match test
4. Monitor progress
5. Final analysis and conclusions

---

**Status:** Test 1 in progress, Test 2 ready to launch
**Updated:** 2026-01-31 17:50
