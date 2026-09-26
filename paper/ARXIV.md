# arXiv submission metadata

Upload `arxiv_submission.tar.gz`, built from this directory. arXiv compiles `main.tex` with pdflatex and uses the included `main.bbl`.

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
