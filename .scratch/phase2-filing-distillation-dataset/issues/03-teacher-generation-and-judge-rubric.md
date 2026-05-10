# 03 - Teacher generation and judge rubric

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 02-filing-example-schema-and-verifier

## Goal

Define how a stronger model generates candidate filing examples and how an LLM judge may be used safely.

## Scope

**In scope:** teacher prompt, JSON contract, cost gate, judge rubric, allowed judge decisions, forbidden judge decisions.

**Out of scope:** running teacher generation.

## Acceptance criteria

- Teacher prompt requires JSON-only output with question, type, cited line items, computation, and final answer.
- Judge rubric is limited to answerability, citation relevance, explanation faithfulness, and format quality.
- Numeric correctness and arithmetic correctness are explicitly programmatic, not judge-only.
- Cost gate includes max sections, max cost, retry policy, and manual review sample.

## Notes / open questions

- The judge can improve filtering but cannot replace the verifier.
