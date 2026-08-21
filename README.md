# MolecularArena training and evaluation scripts

Training, inference, and evaluation code for **MolecularArena: Benchmarking Molecular Learning Models for Multi-Objective Organic Molecule Property Prediction**.

## Included workflows

- Graph models: the core GNN notebook, EdgeCNN, GPSE, and PNA
- Molecular language models: ChemBERTa, MoLFormer, and ChemFormer
- Shared multitask transformer fine-tuning and checkpoint-resume utilities
- Per-model precision, recall, enrichment, and ranking notebooks
- Slurm job scripts used for cluster execution

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the entry points and expected inputs.

## Data and model artifacts

This repository is intentionally code-only. Datasets, trained checkpoints, predictions, logs, plots, papers, and other generated artifacts are not included. The scripts expect the data and model assets described in `REPRODUCIBILITY.md` to be supplied locally.
