# Output Format Examples

## Output Files Generated

After running `./run.sh`, two files are created in the `output/` directory:

### 1. incorrect_items.yaml

Contains ONLY items that have incorrect labels. Format:

```yaml
records:
  - id: 'UgxTzXyCxlnww6zFwCR4AaABAg'
    text: 'ผู้หญิงมันโง่หมดทุกคน'
    predicted_label: 'neutral'
    correct_label: 'GB-ATTACK'
    reason: 'Direct attack on women using derogatory language (โง่ = stupid). This is a clear gender bias attack.'
  
  - id: 'UgxyZgks2H3UdlT9CGN4AaABAg'
    text: 'ผู้หญิงต้องเป็นแม่บ้าน'
    predicted_label: 'neutral'
    correct_label: 'GB-NORMATIVE'
    reason: 'Stereotype enforcing gender role. States women should be homemakers.'
  
  - id: 'UgzmW3OaNb3BLVSnMFd4AaABAg'
    text: 'หีดำแบบนี้มันไม่ใช่ผู้หญิงดี'
    predicted_label: 'GB-ATTACK'
    correct_label: 'GB-SEX'
    reason: 'Using sexual body parts for insult and body-shaming. This is sexualized attack.'

  # ... more incorrect items
```

**Fields:**
- `id` - Record identifier from source
- `text` - The actual text content (Thai)
- `predicted_label` - Label your model predicted (INCORRECT)
- `correct_label` - What the label should be according to guidelines
- `reason` - Why the label was wrong

### 2. summary.yaml

Overall processing statistics:

```yaml
timestamp: '2025-04-11T14:30:45.123456'
total_records: 105114
total_incorrect: 3482
accuracy: '96.68%'
batch_count: 1052
successful_batches: 1050
failed_batches: 2
```

**Fields:**
- `timestamp` - When processing completed (ISO format)
- `total_records` - Total items processed (all 105,114)
- `total_incorrect` - Count of mislabeled items found
- `accuracy` - Percentage of correctly labeled items
- `batch_count` - Total batches processed
- `successful_batches` - Batches validated successfully
- `failed_batches` - Batches that timed out or errored

## Understanding the Output

### Key Points:

1. **Only Incorrect Items in Output**
   - If an item is correctly labeled → NOT in `incorrect_items.yaml`
   - If an item is incorrectly labeled → INCLUDED in `incorrect_items.yaml`
   
2. **Accuracy Calculation**
   - Accuracy = (total_records - total_incorrect) / total_records × 100
   - Example: (105114 - 3482) / 105114 = 96.68%

3. **Label Mapping**
   - Your model predicts one of these: `neutral`, `GB-ATTACK`, `GB-NORMATIVE`, `GB-SEX`, `meta_counter`
   - Gemini API validates against annotation guidelines
   - If prediction doesn't match guidelines → included in output

## Valid Labels Reference

From annotation guidelines (services/auto_annalysis/assets/prompt/annotation/annotation-guideline.md):

| Label | Definition | Examples |
|-------|-----------|----------|
| `neutral` | No gender bias | "ขอบคุณค่ะ" (Thank you), informational statements |
| `GB-ATTACK` | Direct attack on gender/SOGI | "ผู้หญิงมันโง่" (Women are stupid) |
| `GB-NORMATIVE` | Stereotype/gender role | "ผู้ชายต้องเป็นผู้นำ" (Men must be leaders) |
| `GB-SEX` | Sexualized attack/body shame | Using sexual body parts to insult/shame |
| `meta_counter` | Meta commentary/counter-argument | Criticizing society's gender bias |

## Example Analysis

If you see 1000 incorrect items with these patterns:

```yaml
records:
  # Pattern 1: Model thought it was neutral but it's GB-ATTACK
  - predicted_label: 'neutral'
    correct_label: 'GB-ATTACK'
    reason: 'Contains derogatory language attacking gender...'
  
  # Pattern 2: Model thought it was GB-ATTACK but it's meta_counter
  - predicted_label: 'GB-ATTACK'
    correct_label: 'meta_counter'
    reason: 'This is social critique of gender bias, not the bias itself...'
  
  # Pattern 3: Model thought it was neutral but it's GB-NORMATIVE
  - predicted_label: 'neutral'
    correct_label: 'GB-NORMATIVE'
    reason: 'Reflects gender stereotype about women...'
```

**Action Items:**
1. Review patterns in the output
2. Identify where your model is confused
3. Retrain or fine-tune model with corrections
4. Re-run validation to verify improvements

## Checking Processing Progress

During processing, you'll see output like:

```
✓ Batch   001:   100 items,   3 incorrect
✓ Batch   002:   100 items,   1 incorrect
✓ Batch   003:   100 items,   0 incorrect
✓ Batch   004:   100 items,   5 incorrect
...
✗ Batch   156: timeout - Error details
...
✓ Batch 1052:   100 items,   2 incorrect

Processing complete!
Total batches: 1052
Total incorrect items: 3482

✓ Saved 3482 incorrect items to output
✓ Summary saved to: output/summary.yaml
```

## Using the Results

### 1. Review Errors
```bash
# See all incorrect items
cat output/incorrect_items.yaml | less

# Count by error type
grep "correct_label:" output/incorrect_items.yaml | sort | uniq -c
```

### 2. Filter by Pattern
```bash
# See only items predicted as neutral but should be GB-ATTACK
grep -A5 "predicted_label: 'neutral'" output/incorrect_items.yaml | grep -B1 "correct_label: 'GB-ATTACK'"
```

### 3. Get Statistics
```bash
# Count total incorrect
grep "^  - id:" output/incorrect_items.yaml | wc -l

# See first 10 incorrect items
head -100 output/incorrect_items.yaml
```

## Next Steps

1. ✓ Review `output/incorrect_items.yaml`
2. ✓ Identify patterns in errors
3. ✓ Update training data with corrections
4. ✓ Retrain your model
5. ✓ Re-run validation: `./run.sh`

That's it! You now have a detailed analysis of model accuracy and specific errors to fix.
