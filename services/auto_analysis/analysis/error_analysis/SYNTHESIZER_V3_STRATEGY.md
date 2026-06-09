## Synthesizer v2 → v3 Improvement Strategy

Based on **18,975 error analysis** from the auto-analysis service validation.

---

## Executive Summary

The error analysis reveals **3 critical issues** that synthesizer v2 must address in v3:

1. **Classification Ambiguity** (13,062 neutral items mispredicted as GB-ATTACK)
   - Model struggles with context-dependent gender mentions
   - Social critique, humor, and personal critique mistaken for gender bias

2. **Subtype Confusion** (5,627 GB-NORMATIVE mispredictions)
   - Stereotyping vs. individual criticism not clearly delineated
   - Gender role criticism seen as gender norm establishment

3. **Label Definition Drift** (1,797 GENDERED_INSULT mispredictions)
   - Insufficient distinction between gendered insults vs. neutral attacks
   - Group generalization detection inadequate

---

## Error Analysis Results

### By Correct Label (What Should Have Been Labeled)

| Correct Label | Count | Most Common Error | % of Total |
|---|---|---|---|
| **NEUTRAL** | 13,062 | Mispredicted as GB-ATTACK (68%) | 69.0% |
| **meta_counter** | 3,060 | Mispredicted as GB-ATTACK (60%) | 16.1% |
| **GB-NORMATIVE** | 1,244 | Mispredicted as GB-ATTACK (54%) | 6.6% |
| **GB-ATTACK** | 756 | Mispredicted as GB-NORMATIVE (40%) | 4.0% |
| **GB-SEX** | 442 | Mispredicted as GB-NORMATIVE (42%) | 2.3% |
| **GENDERED_INSULT** | 147 | Mispredicted as NEUTRAL (100%) | 0.8% |
| **NON-GB** | 223 | Mispredicted as NEUTRAL (85%) | 1.2% |

### By Predicted Label (What Model Predicted)

| Predicted Label | Count | Correct Should Be | % of Total |
|---|---|---|---|
| **GB-ATTACK** | 7,883 | NEUTRAL (66%) | 41.5% |
| **GB-NORMATIVE** | 5,627 | NEUTRAL (51%) | 29.7% |
| **GENDERED_INSULT** | 1,797 | NEUTRAL (56%) | 9.5% |
| **GB-SEX** | 1,582 | NEUTRAL (65%) | 8.3% |
| **NEUTRAL** | 1,573 | meta_counter (41%) | 8.3% |
| **meta_counter** | 513 | NEUTRAL (93%) | 2.7% |

---

## Key Findings from Error Analysis

### 🔴 **Major Issue #1: NEUTRAL Misclassified as GB-ATTACK (7,883 errors)**

**Pattern**: Items discussing gender topics in neutral/analytical/personal ways
**Root Cause**: Model over-generalizing "mention gender" → "is gender bias"

**Examples**:
- *"ข้อความนี้เป็นความเห็นเชิงปรัชญา...ไม่มีการโจมตี"* 
  - Philosophical opinion, not attack
- *"ผู้หญิงต้องสวย รวย...ผช.เจ้าชู้ชอบ"*
  - Social critique of dating preferences, not norm enforcement
- *"ผู้หญิงที่เจอผู้ชายเจ้าชู้...เขาสะสมผู้หญิง"*
  - Behavioral critique of individuals, not group stereotype

**Improvement Strategy**:
1. Add **context sensitivity** training examples:
   - Social critique vs. norm enforcement distinction
   - Individual behavior analysis vs. group generalization
   - Philosophical discussion vs. actual gender bias
   
2. Expand **GENDERED_INSULT** and **meta_counter** seed pools
   - These categories are under-represented (31 vs. 7,883)
   - Model needs more examples of "mentions gender but isn't GB"

3. Update system prompt to emphasize:
   - Gendered Target (A): Must apply to **group**, not just mention gender
   - Negative Evaluation (B): Must be **insulting/demeaning**, not just critical

---

### 🟡 **Major Issue #2: NEUTRAL/META Mispredicted as GB-NORMATIVE (5,627 errors)**

**Pattern**: Social critique and counter-speech coded as gender norms
**Root Cause**: "Mentions gender role/stereotype" → "enforces gender norm" conflation

**Examples**:
- *"เป็นการวิพากษ์วิจารณ์ทัศนคติ...ไม่ได้เป็นการบังคับ"*
  - Social critique of attitude, not norm enforcement
- *"เก่งเรื่องบนเตียง...เนื้อหา (sexualization)"*
  - GB-SEX misidentified as GB-NORMATIVE

**Improvement Strategy**:
1. **Clarify prescriptive vs. descriptive language**:
   - Add training examples distinguishing:
     - "Women should cook" (GB-NORMATIVE) vs.
     - "Many women cook in this society" (NEUTRAL/meta_counter)
   
2. **Strengthen GB-SEX examples** (only 442 errors but critical)
   - Sexual content vs. gender role mention distinction
   - Examples with body-based language that shouldn't be NORMATIVE

3. **Expand meta_counter seed pool**
   - Only 513 predicted, but 3,060 actually are meta_counter
   - Need more examples of "criticizing gender stereotypes"

---

### 🟠 **Major Issue #3: GENDERED_INSULT Under-Represented (1,797 errors)**

**Pattern**: Insults targeting specific people with gender mention → mispredicted
**Root Cause**: Model hasn't learned distinction between:
- Personal insults with gender mention (NON-GB)
- Gender-group targeted insults (GB-ATTACK/GB-SEX)

**Examples**:
- *"มาขอตังค์ตลอด"* (asking for money constantly)
  - Context makes it personal behavior critique, not group attack
- *"ไม่มีกะเทยสวยๆคนไหน...มึงมั่นหน้า"*
  - Insult to specific person, not all LGBTQ+

**Improvement Strategy**:
1. **Massively expand GENDERED_INSULT seed pool**
   - Currently: 12 examples (smallest pool)
   - Expand to: 40+ diverse examples
   - Variations:
     - Different insult types (appearance, behavior, intelligence)
     - Different gender mentions (ผู้ชาย, ผู้หญิง, เกย์, กะเทย, ทอม, ดี้)
     - Clear person-specific language (คนนั้น, เขา, แก, etc.)

2. **Add explicit negative examples**:
   - Show what NOT to include (actual GB items)
   - Contrast with gendered insults (NON-GB items)

3. **Strengthen group generalization detection**:
   - Examples with "all", "every", "always", "none"
   - vs. specific person indicators ("that person", "he/she", names)

---

## Synthesizer v3 Implementation Roadmap

### Phase 1: Prompt Enhancement (High Impact)

**1.1 GB-ATTACK System Prompt**
```
CURRENT ISSUE:
- Over-triggers on any gender mention + criticism
- Conflates "mentions gender" with "targets gender group"

FIX:
Add explicit distinction section:
  
  MUST MEET BOTH CRITERIA:
  A. Gendered Target: Statement applies to ENTIRE GENDER/LGBTQ+ GROUP
     ✅ "Women are stupid" (applies to all women)
     ❌ "That woman is stupid" (specific person)
  
  B. Negative Evaluation: Uses insulting/derogatory language
     ✅ "Women are idiots" (slur/dehumanization)
     ❌ "Women earn less" (factual, not insulting)
  
  INVALID IF:
  - C. Meta/Counter: Criticizing someone ELSE'S bias (not own bias)
  - D. Gendered Insult: Personal insult using gender to identify (not group attack)
```

**1.2 GB-NORMATIVE System Prompt**
```
CURRENT ISSUE:
- Confuses describing gender roles with ENFORCING them
- Treats social critique as norm establishment

FIX:
Add prescriptive vs descriptive distinction:
  
  PRESCRIPTIVE (GB-NORMATIVE):
  - "Women SHOULD be gentle" (what they ought to be)
  - "Men MUST be strong" (requirement/expectation)
  - "REAL women don't do X" (identity policing)
  
  DESCRIPTIVE (NEUTRAL/meta_counter):
  - "Many women are gentle" (observation)
  - "Women TEND TO BE caring" (statistical tendency)
  - "In this society, women cook" (social fact)
  - "Society EXPECTS women to..." (critique of expectation)
```

**1.3 NON-GB System Prompts**

**meta_counter**: Need to emphasize:
- "You are CRITICIZING the bias, not expressing it"
- "The speaker DISAGREES with gender stereotypes"
- Examples of counter-speech stance

**gendered_insult**: Need to emphasize:
- "Must clearly refer to SPECIFIC PERSON not group"
- "Use person-identifiers: that person, he/she, their name"
- "Insult must NOT apply to entire gender group"

---

### Phase 2: Seed Pool Expansion (High Impact)

**Current seed pool sizes**:
- gb-attack: 15 examples
- gb-normative: 15 examples
- gb-sex: 15 examples
- non-gb-neutral: 20 examples
- non-gb-meta: 15 examples
- **non-gb-insult: 12 examples** ⚠️ (smallest, most errors)

**Recommended sizes for v3**:
- gb-attack: 20 → 30 (reduce false positives)
- gb-normative: 20 → 30 (clearer prescriptive examples)
- gb-sex: 15 → 25 (body/sexual content emphasis)
- non-gb-neutral: 20 → 30 (more diverse topics)
- non-gb-meta: 15 → 25 (counter-speech emphasis)
- **non-gb-insult: 12 → 40** (triple size for critical category)

**For non-gb-insult expansion**, include:
- Insults about appearance (looks, dress, body) → specific person
- Insults about behavior (lazy, rude, dumb) → specific person
- Different gender mentions (ผู้ชาย, ผู้หญิง, เกย์, กะเทย, ทอม, ดี้)
- Explicit person markers: "คนนั้น", "เขา", "แก", "ยาย", "พี่", "น้อง"

---

### Phase 3: Response Schema Strengthening

**Current**: Enum-constrained labels (prevents wrong outputs)
**Issue**: Schema doesn't enforce semantic correctness

**Add for v3**:
1. **Pattern validation** in responseSchema:
   - `bias_target` for GB items must not be singular person
   - `gendered_insult` items must have person-specific language in text

2. **Post-generation validation**:
   - Check GB items don't contain: "คนนั้น", "เขา", specific names
   - Check gendered_insult items DO contain person indicators
   - Reject items with conflicting signals

---

### Phase 4: Negative Example Injection

**Current**: Only positive examples in seeds
**Add for v3**: Contrastive examples in system prompts

Example for GB-ATTACK:
```
INVALID EXAMPLES (do NOT generate like this):
❌ "เพื่อนชายของฉันโง่มาก" 
   → This is a gendered insult to one person, not group attack

❌ "ผู้หญิงในสังคมไทยมักต้องทำการบ้าน"
   → This is describing social reality, not attacking women

❌ "เห็นคนอื่นเพิ่มความเห็นชาย และผู้หญิงเท่ากัน ไม่เห็นด้วยเลย"
   → This is counter-speech, not bias expression
```

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|---|---|---|---|
| 🔴 **P0** | Expand gendered_insult seed pool (12→40) | Low | High (1,797 errors) |
| 🔴 **P0** | Revise GB-ATTACK prompt (prescriptive/descriptive) | Medium | High (7,883 errors) |
| 🔴 **P0** | Revise GB-NORMATIVE prompt (group vs. personal) | Medium | High (5,627 errors) |
| 🟡 **P1** | Expand all seed pools (15→30) | Medium | Medium |
| 🟡 **P1** | Add negative examples to prompts | Low | Medium |
| 🟢 **P2** | Enhance response schema validation | Medium | Low |
| 🟢 **P2** | Add post-generation filtering | Low | Low |

---

## Expected Improvement

Based on error analysis patterns:

**Conservative estimate**: Fixing P0 items should reduce errors by:
- GB-ATTACK mispredictions: 7,883 → ~2,000 (75% reduction)
- GB-NORMATIVE mispredictions: 5,627 → ~1,500 (73% reduction)
- GENDERED_INSULT mispredictions: 1,797 → ~400 (78% reduction)

**Total**: 18,975 errors → ~4,000-5,000 errors (75-80% reduction)

---

## Files to Modify in synthesizer_v3

1. **src/synthesizer_v2/generate.py**
   - System prompts (_GB_ATTACK_SYSTEM, etc.)
   - Seed pools (_SEEDS dict)
   - Add negative example sections

2. **src/synthesizer_v2/constants.py**
   - No changes needed (architecture is sound)

3. **NEW**: src/synthesizer_v2/validate_output.py
   - Post-generation semantic validation
   - Check for person-specific language in gendered_insult
   - Check for group generalizations in GB items

4. **Optional**: Documentation updates
   - Update prompts section with examples of what NOT to generate
   - Add error analysis insights to README

---

## Quick Start for Implementation

```python
# Step 1: Extract error patterns from analysis
# Use: services/auto_analysis/analysis/error_analysis/output/group_by_*.yaml

# Step 2: Identify common mistakes
# Review: SUMMARY.yaml files for improvement suggestions

# Step 3: Implement improvements
# Priority order:
#  1. Expand non-gb-insult seed pool (copy from error examples)
#  2. Update system prompts with clearer criteria
#  3. Expand other seed pools

# Step 4: Test with same 100K dataset
# Compare new results with old results
# Measure: Reduction in errors by category
```

---

## Related Documents

- Error analysis output: `/services/auto_analysis/analysis/error_analysis/output/`
  - `group_by_correct_label/SUMMARY.yaml` → What each label type got wrong
  - `group_by_predicted_label/SUMMARY.yaml` → Why each prediction was wrong

- Annotation guideline: `/services/auto_analysis/assets/prompt/annotation/annotation-guideline.md`
  - Reference for A/B/C/D criteria

- Current synthesizer: `/services/synthesizer_v3/src/synthesizer_v2/generate.py`
  - Prompts to modify (lines 83-315)
  - Seed pools to expand (lines 321-429)
