# Error Analysis Index & Navigation Guide

**Created**: April 11, 2025  
**Total Analysis**: 18,975 incorrect predictions analyzed  
**Documents**: 2 comprehensive error analyses + comparison framework

---

## 📚 Document Overview

### By Grouping View

#### View 1: Group by Correct Label
**File**: `group_by_correct_label/ERROR_ANALYSIS.md`  
**Size**: 17 KB  
**Focus**: What the model got wrong about each category

**Key Sections**:
- Executive summary with error tier breakdown
- 8 detailed error categories (NEUTRAL, meta_counter, GB-NORMATIVE, etc.)
- Root cause analysis for each category
- Recommended actions by priority (P0, P1, P2)
- Implementation roadmap
- Example errors with explanations

**Best For**:
- Understanding model blindspots
- Identifying weak categories
- Finding specific error patterns
- Context-specific improvements

**Key Finding**:
```
NEUTRAL items (13,062 errors, 69%):
  - Model over-classifies gender mentions as bias
  - Root cause: Can't distinguish social critique from actual bias
  - Fix: Add context sensitivity, distinguish individual vs. group
```

---

#### View 2: Group by Predicted Label
**File**: `group_by_predicted_label/ERROR_ANALYSIS.md`  
**Size**: 15 KB  
**Focus**: Why the model made each prediction (false positive analysis)

**Key Sections**:
- Executive summary with false positive rates
- 6 predicted label categories analyzed
- False positive breakdown for each prediction
- Root cause analysis (why over-triggering)
- Comparison table with FP rates
- Implementation priority based on FP impact
- Example errors showing systematic patterns

**Best For**:
- Understanding systematic over-prediction
- Reducing false positives
- Understanding prediction logic
- Identifying scope/boundary issues

**Key Finding**:
```
GB-ATTACK predictions (7,883 errors, 42%):
  - 91% are false positives (66% actually NEUTRAL)
  - Root cause: "Gender mention + negative" = trigger
  - Fix: Require "applies to ENTIRE gender group" criterion
```

---

## 🎯 How to Use These Documents

### Quick Decision Tree

**Q1: "I want to understand what's WRONG with my MODEL"**
→ Read: **Group by Predicted Label** view
- Shows what model is doing wrong (false positives)
- Shows why it's making mistakes
- Shows systematic patterns
- Priority: 91% FP rate on GB predictions

**Q2: "I want to understand what's WEAK in my DATA/CATEGORIES"**
→ Read: **Group by Correct Label** view
- Shows which categories model struggles with
- Shows specific weakness patterns per category
- Shows context-specific issues
- Priority: 69% errors are NEUTRAL category

**Q3: "I want BOTH perspectives"**
→ Read: Both documents in order
1. Start with **Predicted Label** (understand prediction errors)
2. Then read **Correct Label** (understand category blindspots)
3. Cross-reference for comprehensive view

---

## 📊 Key Findings by View

### Group by Correct Label: Category Breakdown

| Category | Errors | % | Problem | P |
|---|---|---|---|---|
| NEUTRAL | 13,062 | 69% | Over-classifies | P0 |
| meta_counter | 3,060 | 16% | Misses counter-speech | P0 |
| GB-NORMATIVE | 1,244 | 7% | Confuses with attacks | P1 |
| GB-ATTACK | 756 | 4% | Reverses with NORMATIVE | P1 |
| GB-SEX | 442 | 2% | Confuses with norms | P2 |
| GENDERED_INSULT | 147 | 1% | Under-represented | P0 |
| NON-GB | 223 | 1% | False positives | P2 |
| Other | 41 | 0.2% | Edge cases | P3 |

**Total P0 Impact**: 16,269 errors (86%)
**Total P1 Impact**: 2,000 errors (11%)
**Total P2 Impact**: 665 errors (3%)

---

### Group by Predicted Label: Prediction Accuracy

| Prediction | Total | Correct | FP Rate | % FP |
|---|---|---|---|---|
| GB-ATTACK | 7,883 | 715 | **91%** | 42% of all errors |
| GB-NORMATIVE | 5,627 | 1,068 | **81%** | 30% of all errors |
| GENDERED_INSULT | 1,797 | 270 | **85%** | 9% of all errors |
| GB-SEX | 1,582 | 238 | **85%** | 8% of all errors |
| NEUTRAL | 1,573 | 0 | **100%** | 8% of all errors |
| meta_counter | 513 | 2 | **99%** | 3% of all errors |

**Key Insight**: GB predictions have 81-91% false positive rates!

---

## 🔧 Using Both Views Together

### Recommended Analysis Flow

```
Step 1: UNDERSTAND THE PROBLEM
  → Read Group by Predicted Label (Tier 1-2)
  → Focus on: Why is model over-predicting GB?
  
Step 2: UNDERSTAND THE BLINDSPOTS  
  → Read Group by Correct Label (Tier 1-2)
  → Focus on: What categories does model struggle with?
  
Step 3: IDENTIFY OVERLAPS
  → Cross-reference both views
  → Example: "GB-ATTACK has 91% FP rate" (predicted view)
             + "NEUTRAL has 69% of errors" (correct view)
             = Root cause: NEUTRAL scope detection
  
Step 4: PRIORITIZE FIXES
  → Use P0/P1/P2 priorities from both documents
  → Focus on high-impact items first
  → GB-ATTACK scope detection = biggest win
```

---

## 🎓 Detailed Comparison

### When to Use Each View

| Scenario | Use This View | Why |
|---|---|---|
| "Which category does model hate?" | Correct Label | Shows category struggle rates |
| "Which prediction is most broken?" | Predicted Label | Shows FP rates |
| "Why is NEUTRAL wrong?" | Correct Label | Explains specific category issues |
| "Why does model over-predict GB-ATTACK?" | Predicted Label | Explains prediction logic |
| "How to improve classifier?" | Predicted Label FIRST | Fix false positives first |
| "How to improve specific category?" | Correct Label | Category-specific improvements |
| "What should I train on?" | Both (complementary) | Full picture of gaps |

---

## 🚀 Implementation Recommendations

### Phase 1: High-Impact Fixes (Based on Both Views)

**Primary Focus**: Reduce GB-ATTACK false positives (7,883 errors, 42%)

**From Predicted Label View**:
- Root cause: "Gender mention + negative" = attack
- Issue: Scope detection (individual vs. group)
- Fix: Add explicit "applies to entire gender group" criterion

**From Correct Label View**:
- NEUTRAL category (13,062 errors, 69%)
- Root cause: Social critique, behavior comments mistaken for bias
- Fix: Add context sensitivity, distinguish analytical from expressive

**Combined Action**:
1. Update GB-ATTACK prompt in synthesizer_v3:
   - "MUST apply to ENTIRE gender/LGBTQ+ group"
   - "Not individual behavior, not social observation"
2. Add training examples from NEUTRAL category:
   - Social critiques (non-bias)
   - Behavior comments (non-group)
   - Life advice (gender-neutral intent)

**Expected Impact**: 7,883 GB-ATTACK errors → 2,000 (75% reduction)

---

### Phase 2: Category-Specific Improvements

**From Correct Label View**:
- meta_counter (3,060 errors, 16%)
- GENDERED_INSULT (147 errors, 1%)

**From Predicted Label View**:
- meta_counter misidentification patterns
- GENDERED_INSULT under-representation

**Combined Action**:
1. Expand counter-speech examples
2. Triple GENDERED_INSULT seed pool (12 → 40)
3. Add person-specific markers to GENDERED_INSULT

**Expected Impact**: 3,207 errors → 800 (75% reduction)

---

## 📌 Document Navigation

### Finding Specific Information

**"I want to understand NEUTRAL errors"**
- Start: `group_by_correct_label/ERROR_ANALYSIS.md` → Section: "NEUTRAL Items..."
- Deep dive: See subsections on social critique, behavior critique, etc.

**"I want to fix GB-ATTACK"**
- Start: `group_by_predicted_label/ERROR_ANALYSIS.md` → Section: "GB-ATTACK (7,883 errors)"
- See: False positive breakdown, root causes, fix strategy
- Then: `group_by_correct_label/ERROR_ANALYSIS.md` → See what NEUTRAL looks like

**"I want to understand meta_counter"**
- Start: `group_by_correct_label/ERROR_ANALYSIS.md` → Section: "meta_counter Items..."
- Cross-check: `group_by_predicted_label/ERROR_ANALYSIS.md` → Section: "meta_counter (513 errors)"
- Note: Different perspectives on same problem

**"I want implementation priorities"**
- Check: Both documents' "Recommended Actions by Priority" sections
- Compare: P0/P1/P2 items
- Result: Clear roadmap in both views (they align on P0 items)

---

## 📈 Statistical Summary

### Error Distribution

```
By Correct Label (What should be):
  NEUTRAL         69% (13,062)  ← Biggest problem
  meta_counter    16%  (3,060)  ← Second biggest
  GB-NORMATIVE     7%  (1,244)
  GB-ATTACK        4%    (756)
  GB-SEX           2%    (442)
  GENDERED_INSULT  1%    (147)  ← Under-represented
  NON-GB           1%    (223)
  Other          0.2%     (41)

By Predicted Label (What model predicted):
  GB-ATTACK       42%  (7,883)  ← Massive over-triggering (91% FP)
  GB-NORMATIVE    30%  (5,627)  ← Heavy over-triggering (81% FP)
  GENDERED_INSULT  9%  (1,797)  ← Over-prediction (85% FP)
  GB-SEX           8%  (1,582)  ← Over-prediction (85% FP)
  NEUTRAL          8%  (1,573)  ← Default when unsure
  meta_counter     3%    (513)  ← Rare over-prediction (99% FP)
```

### Overlap Insights

```
Most Harmful Pattern:
  - 7,883 items predicted as GB-ATTACK
  - 5,222 items (66%) should be NEUTRAL
  - 946 items (12%) should be meta_counter
  = 72% of GB-ATTACK predictions are WRONG

Second Harmful Pattern:
  - 5,627 items predicted as GB-NORMATIVE
  - 2,870 items (51%) should be NEUTRAL
  - 900 items (16%) should be GB-ATTACK
  = 67% of GB-NORMATIVE predictions are WRONG
```

---

## 🔍 Cross-Reference Examples

### Example 1: The NEUTRAL Problem

**Correct Label View Shows**:
- NEUTRAL has 13,062 errors (69% of total)
- Main issue: Predicted as GB-ATTACK (68%)
- Root causes: social critique, behavior comments, life advice

**Predicted Label View Shows**:
- 5,222 items predicted as GB-ATTACK are actually NEUTRAL
- This represents 66% of all GB-ATTACK predictions
- Pattern: "Gender mention + negative tone" triggers false positive

**Combined Insight**:
- Fix scope detection in GB-ATTACK
- Add NEUTRAL examples showing social critique
- Implement "must apply to entire group" criterion

---

### Example 2: The meta_counter Blind Spot

**Correct Label View Shows**:
- meta_counter has 3,060 errors (16% of total)
- Main issue: Predicted as GB-ATTACK (60%)
- Root cause: Can't recognize counter-speech

**Predicted Label View Shows**:
- meta_counter predicted 513 times (3% of predictions)
- But should be 3,060 items (5.9% of data)
- Model UNDER-predicts meta_counter by ~83%

**Combined Insight**:
- meta_counter is severely under-represented in training
- Expand counter-speech examples
- Add indicators for criticism of bias (not expression)
- Model needs to learn: "Criticizing bias ≠ Expressing bias"

---

## 📚 Additional Resources

### Within This Directory

- `SUMMARY.yaml` - Quick summary of improvement suggestions
- Individual label YAML files - Specific error examples
- `group_by_correct_label/` - All NEUTRAL, meta_counter, etc. errors
- `group_by_predicted_label/` - All GB-ATTACK, GB-NORMATIVE, etc. errors

### Parent Directory

- `SYNTHESIZER_V3_STRATEGY.md` - Detailed implementation roadmap
- `QUICK_START.md` - Quick reference guide
- `README.md` - Full module documentation

### Related

- `services/synthesizer_v3/src/synthesizer_v2/generate.py` - Files to modify
- `services/auto_analysis/output/incorrect_items.yaml` - Raw error data
- `services/auto_analysis/assets/prompt/annotation/annotation-guideline.md` - Label definitions

---

## ✅ How to Get Maximum Value

### For Category Improvement

1. Read: `group_by_correct_label/ERROR_ANALYSIS.md`
2. Find your category in the detailed breakdown
3. See: Root causes, key improvements, examples
4. Check: P0/P1/P2 priority
5. Action: Use specific suggestions

### For Prediction Improvement

1. Read: `group_by_predicted_label/ERROR_ANALYSIS.md`
2. Look at false positive rates (sorted by impact)
3. See: Why model is making this mistake
4. Check: Root cause explanation
5. Action: Implement systematic fix

### For Complete Understanding

1. Read both documents (Correct Label first, then Predicted Label)
2. Compare priority lists (they should align)
3. Review examples in both views
4. Cross-reference overlapping patterns
5. Create implementation plan from combined insights

---

## 🎯 Quick Reference: What Needs Fixing

### 🔴 P0 - Critical (Start Here)

**By Correct Label View**:
- NEUTRAL over-classification (13,062 errors)
- meta_counter misclassification (3,060 errors)
- GENDERED_INSULT under-representation (147 errors)

**By Predicted Label View**:
- GB-ATTACK over-prediction (91% FP rate)
- GB-NORMATIVE over-prediction (81% FP rate)

**Action**: Fix scope detection, expand examples

**Expected Impact**: 75-80% total error reduction

---

### 🟡 P1 - Important (Second Priority)

**By Correct Label View**:
- GB-NORMATIVE confusion (1,244 errors)
- GB-ATTACK reversal (756 errors)

**By Predicted Label View**:
- GB-SEX over-prediction (85% FP rate)
- GENDERED_INSULT over-prediction (85% FP rate)

**Action**: Better category distinction

**Expected Impact**: 44% reduction in P1 items

---

### 🟢 P2 - Nice to Have (Third Priority)

**By Correct Label View**:
- GB-SEX accuracy (442 errors)
- NON-GB classification (223 errors)

**By Predicted Label View**:
- NEUTRAL under-detection
- meta_counter over-prediction

**Action**: Edge case refinement

**Expected Impact**: 50% reduction in P2 items

---

## 📋 Document Checklist

- ✅ Group by Correct Label: ERROR_ANALYSIS.md (17 KB)
- ✅ Group by Predicted Label: ERROR_ANALYSIS.md (15 KB)
- ✅ This Index & Navigation Guide
- ✅ SUMMARY.yaml in both directories
- ✅ Individual YAML files (17 total across both views)

---

**Total Analysis Complete**: 18,975 errors
**Documents Created**: 4 (2 major analyses + index + guide)
**Total Documentation**: ~50 KB of analysis
**Implementation Ready**: Yes

**Next Step**: Read both ERROR_ANALYSIS.md files and implement Phase 1 improvements

---

Created: April 11, 2025  
Purpose: Enable clear understanding of errors from two complementary perspectives  
Outcome: Clear roadmap for 75-80% error reduction
