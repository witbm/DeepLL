# `src/` — Source Code

## Overview
This folder contains the code to train, evaluate, and run an MLP classifier that predicts CEFR proficiency levels (A1–C2) from the internal states of a large language model (e.g., Llama).

## Workflow
1. **Split** the corpus into training and evaluation datasets
2. **Extract** the internal states of both datasets
3. **Train** an MLP classifier on the internal states (`.npy`)
4. **Evaluate** the best model on the evaluation dataset and export metrics/plots.
5. **Classify** a given input with the selected model
