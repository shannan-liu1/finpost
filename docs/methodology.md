# Methodology

FinPost is a post-training research harness for testing which signals improve FinChain financial reasoning under constrained compute. The project is intentionally diagnostic: parseability, exact final-answer accuracy, checkpoint selection, and failure analysis matter more than a single headline metric.

The engineering target is reproducibility under weak-signal post-training: each
method is tied to a data split, verifier, selection rule, held-out test report,
and artifact trail.

## SFT

Supervised fine-tuning teaches the model to imitate target reasoning traces and answer formats. In this repo it is used as the baseline adaptation step before preference optimization or reinforcement learning. The key risk is that SFT can improve formatting without improving the model's underlying arithmetic or financial reasoning.

## DPO

Direct Preference Optimization trains from chosen/rejected pairs. FinPost builds preference data from verifier outcomes and uses DPO to ask whether preference learning can improve over a chosen policy checkpoint. In the public FinChain run, that checkpoint is the validation-selected SFT model used as both the pair source and the DPO initialization/reference. DPO quality depends on pair quality; if the sampled policy rarely produces correct candidates, the preference signal can be weak or misleading.

## GRPO / RLVR

Group Relative Policy Optimization is used as a verifier-reward RLVR method. The model samples multiple completions for the same prompt, receives deterministic final-answer rewards, and updates from group-relative advantages. This can work only when the group contains meaningful reward variation. If the model's priors are too weak, most completions are uniformly wrong and the reward signal collapses.

## OPD / GKD

On-policy distillation, implemented here as Generalized Knowledge Distillation, samples student trajectories and trains against a frozen teacher distribution on those trajectories. It tests whether teacher-guided signal is more useful than sparse final-answer reward when the student is weak. The method is compute-heavier because it needs both student and teacher scoring.

## Verifier-Based Reward And Eval

FinPost uses deterministic parsing and final-answer scoring for FinChain outputs. The verifier is intentionally simple: parse a final answer, normalize numeric/categorical forms, and compare against the gold answer. This makes reward and evaluation reproducible and inspectable.

## Why Answer Parsing Matters

A model can show plausible reasoning while omitting a scoreable final answer, or it can produce a correct value in an unparseable format. Parseability is therefore tracked separately from final-answer correctness. This avoids confusing formatting failures with reasoning failures.

## Weak Priors And Post-Training Failures

Small models can fail because their base distribution does not produce enough correct or near-correct reasoning attempts. In that regime, SFT may teach surface imitation, DPO may learn from poor preference pairs, and GRPO may see sparse all-zero rewards. FinPost diagnoses this by checking validation-selected checkpoints, held-out test performance, parseability, and whether downstream methods beat the SFT baseline before treating them as improvements. In the FinChain v2 run, this showed up directly in DPO: the SFT policy was not strong enough to generate informative preference pairs, and the resulting DPO checkpoint regressed below the SFT baseline on held-out test.
