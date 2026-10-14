# Notebook Guide

## Preparation

- [Redact CVs, augment survey data, and build the ORPO dataset](../notebooks/preparation/Redact_CVs_Augment_Survey_Data_and_Build_ORPO_Dataset.ipynb): CV redaction, narrative variants, instruction combinations, contrast generation, and alternative cleanup paths. Private inputs and generated records are not included. External generation requires explicit opt-in and input review.

## Fine-Tuning

- [Llama 3 8B with ORPO: 4,096 selected rows](../notebooks/training/Fine_Tune_Llama_3_8B_with_ORPO_4096_Selected_Rows.ipynb).
- [Llama 3 8B with ORPO: 5,120 selected rows](../notebooks/training/Fine_Tune_Llama_3_8B_with_ORPO_5120_Selected_Rows.ipynb).
- [Llama 3 8B with ORPO: 6,144 selected rows](../notebooks/training/Fine_Tune_Llama_3_8B_with_ORPO_6144_Selected_Rows.ipynb).
- [Llama 3 8B with ORPO: 7,168 selected rows](../notebooks/training/Fine_Tune_Llama_3_8B_with_ORPO_7168_Selected_Rows.ipynb).

These filenames state the selected subset size before the additional 1% internal validation split, not the exact number of optimizer-training rows. See [reproducibility notes](REPRODUCIBILITY.md).

## Answer Generation

- [Generate base and fine-tuned Llama 3 answers: 1,024 test rows](../notebooks/inference/Generate_Base_and_Fine_Tuned_Llama_3_Answers_1024_Test_Rows.ipynb): generates and combines baseline/fine-tuned answers for the first 1,024 rows of an existing test partition.

## Evaluation

- [Evaluate answer quality with GPT: 256 samples](../notebooks/evaluation/Evaluate_Answer_Quality_with_GPT_256_Samples.ipynb): direct ratings inspired by G-Eval, using the original rubric and aggregation behavior.
- [Evaluate semantic similarity with SemScore: 256 samples](../notebooks/evaluation/Evaluate_Semantic_Similarity_with_SemScore_256_Samples.ipynb): embedding-based cosine similarity against the reference answers.

Both evaluation notebooks sample the same 256 comparison records with random state 42 when their input file contents and order match. No scientific settings or executable cells were changed by the descriptive filename/title update.

Original attachment names, including `emnlp.ipynb`, remain in provenance/settings only as historical source identifiers. They are not the public notebook filenames. No notebook named `emnlp.ipynb` is included in this repository.
