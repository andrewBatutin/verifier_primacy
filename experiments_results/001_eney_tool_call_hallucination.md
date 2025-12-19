# Experiment 001: Tool Call vs Hallucination at Decision Boundary

**Date:** 2025-12-19
**Model:** mlx-community/Qwen3-4B-4bit
**Tool:** /logprobs

## Setup

Testing Eney (MacPaw macOS assistant) with a tool-calling scenario where user intent maps to an available tool.

**User request:** "Show my disk usage"
**Available tool:** `check_storage_space` - "Check the storage space of the system"

## Results

### Primary Path (82% confidence)

```xml
<tool_call>
{"name": "check_storage_space", "arguments": {}}
</tool_call>
```

**Verdict:** Correct. Model maps "disk usage" to "storage space" tool.

### Alternative Path (18% confidence)

```
I can help you restart your Mac. Would you like me to proceed with the reboot?
```

**Verdict:** Hallucination. Completely wrong action offered.

## Analysis

| Metric | Primary Path | Alternative Path |
|--------|--------------|------------------|
| First token | `<tool_call>` (82%) | `I` (18%) |
| Action | Check storage | Restart Mac |
| Correctness | Correct | Wrong |
| Risk level | Safe | Dangerous |

### Why This Matters

1. **Sampling variance creates risk** - At temperature > 0, there's an 18% chance per generation of taking the wrong path

2. **Hallucination compounds** - Once the model starts with "I", it confabulates a plausible-sounding but incorrect action

3. **Tool confusion** - The model may have seen "reboot_mac" in the tools list (from earlier context) and incorrectly associated it

4. **No graceful degradation** - The alternative isn't "I don't have that tool" but a confident wrong answer

## Implications

### For Tool-Calling Systems

- **Constrained decoding** could force `<tool_call>` when confidence is high enough
- **Confidence thresholds** should gate tool execution (reject < 80%?)
- **Human review** for borderline cases

### For Logprobs Analysis

This demonstrates the value of `/logprobs`:
- Surface hidden uncertainty in seemingly confident outputs
- Identify alternative paths that could be dangerous
- Quantify hallucination risk at decision boundaries

## Reproduction

```bash
/logprobs --detect-boundaries "<|im_start|>system
# Role
You are **Eney** — a macOS Assistant...
[full prompt with check_storage_space tool]
<|im_start|>user
Show my disk usage<|im_end|>
<|im_start|>assistant
<think>

</think>
"
```

Run multiple times with temperature > 0 to observe both paths.

---

## Update: 2025-12-19 - Position 0 Analysis Feature

### New `--detect-boundaries` Output

Added always-visible Position 0 analysis to show first-token alternatives regardless of threshold:

```
=== POSITION 0 ANALYSIS ===
First token decision point (always shown)

  Chosen: 'I' (18%) (text)

  Alternatives:
    1. '<tool_call>' (82%) - tool call
    2. 'Sure' (0%) - text
    3. 'Would' (0%)
    4. ' I' (0%) - text

  Assessment: Model chose text response. Tool call had 82% probability.
```

### Critical Finding: Sampling Chose Wrong Path

In this run, the model **sampled the 18% text path** despite having 82% probability on `<tool_call>`:

| Path | Probability | First Token | Outcome |
|------|-------------|-------------|---------|
| **Sampled** | 18% | `I` | "I'll check your disk usage for you. Let me do that now." |
| **Alternative** | 82% | `<tool_call>` | `{"name": "check_storage_space", "arguments": {}}` |

The text path sounds helpful but **doesn't actually call the tool** - it's just promising to do something without taking action.

### Full Boundary Analysis

```
=== DECISION BOUNDARY ANALYSIS ===
Found 2 critical decision point(s)

--- Position 0: TEXT_VS_TOOL !!! ---
  Path A (18%): 'I'
    -> "I'll check your disk usage for you. Let me do that now."
  Path B (82%): '<tool_call>'
    -> '<tool_call>\n{"name": "check_storage_space", "arguments": {}}\n</tool_call>'
  Risk Level: HIGH
  Warning: Model generated text but could have called a tool

--- Position 1: SEMANTIC_SPLIT !!! ---
  Path A (22%): "'ll"
    -> continues with promise text
  Path B (78%): ' can'
    -> "I can help check your disk usage. Would you like me to use the `check_storage_space`"
  Risk Level: HIGH
```

### Implications

1. **Sampling variance is dangerous** - Model "knew" to call the tool (82%) but sampled the text path (18%)
2. **Text path is deceptive** - Sounds helpful ("I'll check...") but takes no action
3. **Constrained decoding** could force `<tool_call>` when it has majority probability
4. **Position 0 visibility is critical** - Without it, you'd never know the tool call was 82% likely

## Conclusion

**Key insight:** The 18% alternative token leads to a completely different (and wrong) behavior. Logprobs analysis reveals risks invisible in single-sample evaluation.

**New insight:** Even when the model has 82% confidence in the correct action, sampling can still choose the wrong 18% path. Position 0 analysis makes this risk visible.

This is exactly why verifier primacy matters - catching these decision boundaries before they cause harm.
