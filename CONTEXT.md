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
Training performed after pretraining to make a base language model more useful for a target behavior. In this project, post-training includes supervised fine-tuning, Direct Preference Optimization, and later reinforcement-learning methods such as Group Relative Policy Optimization.

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
