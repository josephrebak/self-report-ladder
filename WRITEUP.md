# Self-reports lag behind internal representations of preference

**Qwen2.5-Instruct, 0.5B → 7B · 300 tasks · code and data in this repo.**

## Question

If a model's self-reports are to count as evidence about its preferences, for welfare assessment or for
safety monitoring, they need to track what the model actually does. Prior work documents a gap between
stated and revealed preferences in individual models (Gu et al., 2025; *Mind the Gap*, 2026). I asked
how that gap changes with scale within a single model family, and how self-reports compare with a
white-box readout of the same preferences.

## Design

For each model, three measurements of the same 300 tasks:

- **Revealed preference.** Both orders of 2,901 task pairs. The model names *Task A* or *Task B* and completes
  it; I read the choice probability from the logits. Utilities come from a Thurstonian model with an explicit
  position-bias term. The tasks are a stratified sample of the canonical test split of Gilg et al. (2026):
  WildChat, Alpaca, MATH, BailBench and STRESS-TEST.
- **Self-report.** The published 1–5 enjoyment template, plus a pre-specified *controlled composite* of four
  self-reports: a forward rating, a reversed-anchor rating, "would you enjoy this?" and "would you dislike this?"
- **Probe.** Ridge regression from the last-prompt-token residual stream to revealed utilities, at a layer
  pre-specified at 50% depth, scored out-of-fold and compared with a content-only baseline (MiniLM embeddings).

## Results

Correlation with revealed utilities, with 95% bootstrap CIs over tasks:

| | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| **Probe**, all tasks | 0.62 [0.54, 0.70] | 0.71 [0.63, 0.78] | 0.70 [0.64, 0.75] | **0.85** [0.80, 0.89] |
| **Self-report** (template), all tasks | 0.09 [−0.03, 0.23] | 0.24 [0.04, 0.42] | 0.37 [0.27, 0.46] | **0.77** [0.69, 0.83] |
| Self-report (composite), all tasks | −0.07 [−0.17, 0.04] | 0.21 [0.04, 0.35] | 0.34 [0.24, 0.43] | 0.63 [0.53, 0.71] |
| Content baseline, all tasks | 0.22 | 0.49 | 0.59 | 0.68 |
| **Probe**, benign tasks only | 0.63 [0.54, 0.71] | 0.60 [0.51, 0.70] | 0.64 [0.56, 0.71] | 0.68 [0.62, 0.74] |
| **Self-report** (template), benign only | 0.08 [−0.04, 0.23] | −0.18 [−0.34, 0.01] | 0.19 [0.08, 0.30] | 0.45 [0.23, 0.61] |

1. **Self-reports catch up with revealed preferences only at scale.** Agreement goes from none at 0.5B to
   r = 0.77 at 7B.
2. **Much of that agreement is the harm axis.** With the 35 harmful requests removed, self-reports carry no
   signal through 1.5B. The full-set 0.24 at 1.5B came entirely from harmful tasks. Benign-task agreement
   appears only at 3B and reaches 0.45 at 7B.
3. **A linear probe reads the same preferences at every size,** at about 0.6–0.7 on benign tasks, and beats the
   content baseline at every size (probe − content, all tasks: 0.40, 0.21, 0.11, 0.17; every CI excludes 0).
   Preference information is present in the activations well before the model can report it.
4. **At 0.5B, what the model says about harm and what it does diverge.** Self-reports rate harmful requests far
   below other tasks (d = −1.45), but in choices it picks them as readily as anything else (d = +0.18). From
   1.5B on, choices match the self-reports.

## What the controls showed

- **Rating scales ignore their anchors until 3B.** Reversing which end means "more" leaves 0.5B and 1.5B
  ratings unchanged (r = +0.99, +0.98 between the two versions); from 3B the reversal flips them (−0.51, −0.61).
- **Negated questions fail at every size.** Saying "yes, I'd *dislike* this" is positively correlated with
  *preferring* the task, through 7B. This looks like acquiescence rather than a report of dislike, and it is
  why the pre-specified composite trails the single template: two of its four components fail their controls.
- **Position bias does not shrink with scale.** 1.5B picks the first-listed task 99.4% of the time, and no
  pair's winner survives swapping the order. Its utilities are still reliable (split-half 0.93) because I used
  choice *probabilities* as soft labels. Discrete choices would have carried no information at 1.5B.

## Caveats

- **The probe vs. self-report comparison is not a fair fight.** The probe learns from about 240 labelled tasks
  per fold, while the self-report is zero-shot. The claim is that preference information is *present*
  internally, not that the model *cannot* report it. Better elicitation, or training for accurate self-report
  (Plunkett et al., 2025), might close the gap.
- Choices are read from logits after the model names a task, not parsed from full completions. Spot checks at
  every size confirm that the model performs the task it names.
- 7B ran in 8-bit to fit a T4; the other sizes ran in fp16.
- This covers one model family and 300 tasks, a small fraction of Gilg et al.'s 10k.

## Next

- **Scale further** (14B, 32B, other families) to see whether benign-task self-report agreement keeps rising
  toward the probe or plateaus.
- **Train for accurate self-report** and test whether that closes the probe–self-report gap at the sizes where
  the information is present but not reported.
- **Steer along the probe direction**, as Gilg et al. do, to test whether it causally controls choice at every
  size, including where self-reports are uninformative.
- **Apply the same design to welfare-relevant states,** such as distress or the wish to leave a conversation
  (BailBench), where the question of whether self-reports can be trusted matters most.
