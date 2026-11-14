# arXiv submission metadata

Upload `arxiv_submission.tar.gz`, built from this directory. arXiv compiles `main.tex` with pdflatex and uses the included `main.bbl`.

## v2 (replacement, draft)

Submit as a replacement of the v1 identifier once v1 is announced, not as a new submission. Rebuild the tarball from the v2 `main.tex`, `main.bbl`, `tables/` and `figures/`.

**Abstract (v2)**
System-One models such as Jev answer typed questions about a state with calibrated probabilities over a predefined set of answers, in one fast pass. Recent decision models add test-time compute by reasoning in text when unsure, at a cost of seconds per decision. We study spending it in latent space instead. z-reason keeps the typed interface and adds a steps parameter: a weight-tied block iterates over one vector per fact of the state for the requested number of steps, and training samples the number of steps at random. On symbolic variable tracking, a model with 4.1M parameters trained on chains of at most 8 hops answers 16 hops with at least 99% accuracy. In a natural-language version built on frozen sentence vectors from an open decision model, accuracy with the wording seen in training rises from about 50% with one step to at least 98% with 8 steps up to three times the training depth, while a one-pass model stays near a shortcut baseline. With held-out wording it reaches only 62%. On ProofWriter, trained on proofs of depth at most 3, z-reason answers depths 1 to 3 with 96% to 97% accuracy against 70% to 78% for one pass, and a new domain with 93% against 79%. Its True and False answers are right 98% of the time, but on deeper proofs it answers Unknown more and more often and falls to 37% at depth 5, where the one-pass model scores 77% through shortcuts that fail on Unknown questions. Sixteen steps take 41 to 61 ms per request on a consumer GPU. Latent steps give cheap and sound composition within the training depth, with limited extrapolation on ProofWriter and partial robustness to wording.

**Comments (v2)**
18 pages, 8 figures, 13 tables. v2 adds natural-language variable tracking, ProofWriter and latency measurements. Code: https://github.com/silvaan/z-reason

## v1

**Title**
Latent Reasoning Steps for Typed Decisions: Test-Time Compute for System-One Models

**Authors**
Silvan Ferreira, Thiago Medeiros, Ivanovitch Silva

**Abstract**
System-One models such as Jev answer typed questions about a state with calibrated probabilities over a predefined set of answers instead of generating text. Their output is type safe by construction and one fast pass answers every question, but easy and hard questions receive the same computation, so these models cannot benefit from test-time compute. We present z-reason, a small model with the same request and response interface that adds a steps parameter. Between encoding and read-out it iterates a weight-tied block over a latent state z for the requested number of steps, much as a diffusion sampler runs a chosen number of denoising steps. The recurrence operates on one vector per fact of the state, and the number of steps is sampled at random during training. On a synthetic variable tracking task whose difficulty is the length of a dependency chain, a model with 4.1M parameters trained on chains of at most 8 hops answers chains of up to 16 hops with at least 99% accuracy given 8 or more steps, in each of three seeds, while extrapolation to three or four times the training depth is partial and seed dependent. A transformer without weight tying and with 15.1M parameters falls close to a root guessing baseline beyond its training range. The number of steps needed grows sublinearly with depth, random step training keeps extra steps from hurting accuracy where training with a fixed number of steps does not, calibration error falls as steps increase, and a convergence rule for halting spends more steps on harder questions. All results come from a single synthetic task, so they show the mechanism in a controlled setting and do not yet speak to decisions over natural language.

**Comments**
13 pages, 6 figures, 7 tables. Code: https://github.com/silvaan/z-reason

**Primary category**
cs.LG

**Cross-lists**
cs.AI, cs.CL

**License**
CC BY 4.0
