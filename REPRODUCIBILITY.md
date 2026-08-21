# MolecularArena reproducibility guide

Run commands from the repository root. The training and test data are available as `Training_and_Eval_Dataset.csv` and `Test_dataset.csv`, with mirrored copies under `Dataset/`.

## Model entry points

- Core graph workflow: open and execute `gnn_yinqi.ipynb`.
- EdgeCNN: `python EdgeCNN/EdgeCNN_infer_eval_updated.py`
- PNA: open and execute `PNA/PNA_training_evaluation.ipynb`.
- GPSE: `python GPSE/GPSE_infer_eval_python-update.py`
- ChemBERTa: `python chemberta/train_chemberta.py`
- MoLFormer: `python molformer/train_molformer.py`
- ChemFormer: `python chemformer/train_chemformer.py`

The language-model scripts share the training utilities in `multitask_transformer_finetune.py`. ChemFormer additionally reads `chemformer/hparams.yaml`, the supplied vocabulary files, and the supplied checkpoint.

## Evaluation artifacts

Per-model prediction CSV files are stored with their corresponding model directories. Top-100 precision, recall, enrichment, normalized scores, and ranked molecule lists are under `precision_recall_enrichment/`; each model directory contains the calculation notebook and generated CSV outputs.

The paper and Appendix F document the fixed data split, objectives, masking rules, model settings, optimizers, learning rates, batch sizes, epoch counts, schedulers, sequence lengths, and random seeds used for the reported experiments.
