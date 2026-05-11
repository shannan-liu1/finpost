# Context: finpost

A learning project to internalize the fundamentals of language-model post-training by applying them to financial data drawn from EDGAR filings. The financial domain is a forcing function for non-toy data, not a product target.

## Project intent

- **Primary goal (~70%):** Learn the fundamentals of post-training at a deep level — including the working vocabulary used by practitioners. No abbreviations in documentation; spell out terms on first use.
- **Secondary goal (~30%):** Produce a credible artifact in the financial-reasoning domain that reflects the user's background and interest in finance, distinguishing this work from a generic benchmark project.
- **Success bar:** The evaluation harness shows a statistically defensible improvement on a clearly-scoped problem the user defined. Not: "an analyst would use this model."

## Target capability

The model is being post-trained to perform two related skills over a single filing excerpt provided as input context:

1. **Numerical extraction** — given an excerpt, return a specific figure asked about (e.g. "What was operating income for fiscal year 2023?"). Verified by exact numeric or string match against ground truth.
2. **Numerical reasoning over a filing** — given an excerpt, *compute* a derived quantity (year-over-year change, gross margin, segment contribution to total revenue). The model must identify the correct line items *and* perform the arithmetic correctly. Verified programmatically against ground truth.

Explicitly out of scope for this project:
- Open-ended analytical answers requiring qualitative judgment
- Accounting-textbook problems (journal entries, statement preparation) divorced from a real filing
- Cross-filing retrieval, reconciliation, or comparison

## Glossary

### Filing
A document submitted by a registrant to the United States Securities and Exchange Commission via the EDGAR system. In this project, restricted to annual reports (Form 10-K) and quarterly reports (Form 10-Q).

### Filing excerpt
A bounded slice of a filing (one section, one table, or a small group of related paragraphs) provided to the model as input context for a single task. The unit of grounding — every numerical answer must be traceable to the excerpt that was given.

### Numerical extraction
The task of returning a single figure that appears verbatim in a filing excerpt. Distinguished from numerical reasoning by the absence of any arithmetic on the model's part.

### Numerical reasoning
The task of computing a derived quantity from one or more figures present in a filing excerpt. Requires both correct selection of input line items and correct arithmetic.

### Post-training
Training performed after pretraining to make a base language model more useful for a target behavior. In this project, post-training includes supervised fine-tuning, On-Policy Distillation, Direct Preference Optimization, and later reinforcement-learning methods such as Group Relative Policy Optimization.

### Supervised Fine-Tuning
A post-training method that updates a model on prompt-response examples. The model is taught to imitate the provided response tokens, usually with loss applied only to the assistant response and not to the prompt. Abbreviated SFT after first use.

### Direct Preference Optimization
A post-training method that updates a model from pairs of responses to the same prompt: one preferred response and one non-preferred response. It is used when the target behavior is easier to express as "response A is better than response B" than as one exact gold answer. Abbreviated DPO after first use.

### Group Relative Policy Optimization
A reinforcement-learning post-training method that samples multiple responses for the same prompt, scores each response with a reward function, and updates the policy based on each response's relative standing within the group. Abbreviated GRPO after first use.

### Teacher distillation
Using a larger or stronger model to produce training examples, labels, rationales, preference pairs, or grading signals for a smaller student model. In this project, teacher distillation is a data-generation technique, not a guarantee that the teacher output is true.

### LLM as judge
Using a language model to evaluate or compare generated answers. In this project, LLM-as-judge can help score faithfulness, citation quality, and answerability, but numerical answers and computations still require programmatic verification.

### On-Policy Distillation
A post-training method that updates a model on preference pairs sampled from the current policy's own generations. For each prompt, the policy generates multiple completions; a programmatic verifier labels each as correct or incorrect; preferred (correct) and non-preferred (incorrect) completions are paired and fed through a Direct-Preference-Optimization-style pairwise loss. Distinguished from offline Direct Preference Optimization by the fact that the pair distribution tracks the current model rather than a fixed dataset. Abbreviated OPD after first use.

### Compute-aware post-training
A workflow philosophy that treats compute as an experimental variable rather than a sunk cost. Every post-training method is reported alongside its rollout cost, verifier cost, and training cost — measured in tokens, GPU-hours, and dollars — and improvements are scored per dollar and per GPU-hour, not only by accuracy. Inspired by the llm.c reproduction of GPT-2 (10B FineWeb tokens, ~90 minutes on 8×A100, ~$20).

### Rollout
A batch of completions generated by a policy model for a fixed set of prompts. The unit of generation cost in this project. A rollout is parameterised by `(prompts, samples_per_prompt K, max_output_tokens, model_size)`. Rollout output is cached on disk so that downstream verification, bucketing, and preference-pair construction never trigger a regeneration unless the policy or sampling parameters change.

### Verifier
A function that takes a single model completion and returns a binary or graded correctness signal. In this project, verifiers are ordered cheapest-first — exact answer string match, symbolic or numeric equivalence, unit tests for code, then small local verifier model, then LLM-as-judge only for unresolved cases. The verifier is run on every completion in every rollout; verifier calls are counted as a first-class cost metric.

### Difficulty bucket
A coarse label assigned to a prompt based on the fraction of K rollout samples that the verifier marks correct. In this project the default buckets are `easy` (p_correct ≥ 0.8), `ambiguous` (0.2 ≤ p_correct ≤ 0.8), and `hard` (p_correct < 0.2). The bucket controls how much additional rollout compute and how much preference-update weight a prompt receives.

### Adaptive sampling
A rollout strategy that draws a small initial number of samples per prompt (K=4), assigns a difficulty bucket, and then draws additional samples (e.g. K=12–28 extra) only for ambiguous prompts. Reduces total rollout tokens versus a uniform K while concentrating signal where the model is uncertain.

### Cost ledger
A per-run record that captures, in addition to accuracy, the rollout tokens generated, the number of verifier calls, the training tokens consumed, the wall-clock GPU-hours, the estimated dollar cost, and the derived ratios `accuracy / dollar` and `accuracy / GPU-hour`. The cost ledger is the headline reporting surface for compute-aware post-training.
