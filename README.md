# Alumni Entrepreneurship and Innovation: LLM Study Code

Code supporting the manuscript **Assessing Alumni Entrepreneurship and Innovation Trajectories through Large Language Models**.

This repository contains cleaned copies of the supplied training, inference, and scoring implementations. It is a partial reproducibility resource, not a fully reconstructed or execution-validated historical environment. No participant records, generated comparison answers, scored datasets, CVs, credentials, or model weights are included.

## Workflow

| Stage | Notebook | Inputs |
|---|---|---|
| ORPO training | [4,096 selected rows](notebooks/training/ORPO_Training_4096.ipynb), [5,120](notebooks/training/ORPO_Training_5120.ipynb), [6,144](notebooks/training/ORPO_Training_6144.ipynb), [7,168](notebooks/training/ORPO_Training_7168.ipynb) | Authorized preference dataset and base-model access |
| Answer generation | [Inference comparison](notebooks/inference/Inference_Comparison_1024.ipynb) | Existing test partition and four fine-tuned model locations |
| Direct-rating evaluation | [GPT scorer](notebooks/evaluation/Evaluation_GEval_GPT_256.ipynb) | Authorized local comparison CSV and explicitly enabled API access |
| Semantic similarity | [SemScore scorer](notebooks/evaluation/Evaluation_SemScore_256.ipynb) | Authorized local comparison CSV and embedding-model access |

The baseline is `meta-llama/Meta-Llama-3-8B-Instruct`. Each training notebook selects its named number of rows from the existing training partition and reserves an additional 1% for internal validation. Inference selects the first 1,024 rows of the existing test partition. Both scorers sample 256 comparison rows with random state 42.

The original augmentation/anonymization implementation, rejected-answer-generation requests, and upstream partition-creation code were not recovered. This repository does not create or independently certify the original train/test split. See [reproducibility notes](docs/REPRODUCIBILITY.md) for scope, retained implementation limitations, and historical environment gaps.

## Environment And Inputs

The historical study used Google Colab with CUDA GPU resources. GPU notebooks require suitable hardware, compatible dependencies, and authorization to access the specified base model and the caller's study inputs. The original dependency declarations are in [requirements](requirements/); several package versions were unpinned. These files are not tested modern installation recipes or complete historical locks. Full GPU/API execution has not been tested during repository preparation.

[`.env.example`](.env.example) lists the environment variables. The notebooks do not automatically load that file. Set the variables in the execution environment before running the cells.

- `CAEE_DATASET_REPO`: an authorized Hugging Face dataset repository exposing the relevant `train` and/or `test` partitions.
- `HF_TOKEN`: optional authentication supplied securely through the environment. Never commit it.
- `CAEE_FINETUNED_MODELS`: a JSON object mapping `4096`, `4096x1-25`, `4096x1-50`, and `4096x1-75` to authorized model repositories or local merged-model directories.
- `CAEE_EVALUATION_CSV`: an authorized local comparison CSV produced by inference.
- `CAEE_INFERENCE_OUTPUT_DIR` and `CAEE_EVALUATION_OUTPUT_DIR`: local output directories. Keep generated artifacts out of Git.

Training inputs must contain `prompt`, `chosen`, and `rejected`; the preference fields are role/content message sequences. Inference additionally requires `context` and `question`, with the reference answer in the final assistant message of `chosen`. Comparison inputs require `question`, `expected_answer`, and `parsed_answer_` columns for `base` and the four fine-tuned suffixes. No compatible study input is bundled.

## External Scoring

The GPT notebook sends questions, reference answers, and generated answers to an external API and may incur charges. It requires both `OPENAI_API_KEY` supplied securely and explicit `CAEE_ALLOW_EXTERNAL_EVALUATION=1`. It is disabled by default. Use only records authorized for that processing; running from an institutional cloud account is not a substitute for checking input permissions.

Historical model identifiers are retained for transparency. Availability of those models or APIs and compatibility with current package releases is not guaranteed. The original parser/averaging behavior is preserved and documented, not silently replaced with a different scientific method.

## Validation Without Research Data

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

These standard-library tests check notebook syntax and cleanliness, settings, generic input/output parsing, external-scoring opt-in, and file-manifest integrity. They use synthetic strings and stubbed responses only. They do not train models, download datasets, call external APIs, or reproduce the reported scientific scores.

## Provenance And Availability

[`PROVENANCE.json`](PROVENANCE.json) identifies the original notebook filenames and hashes and records changes made to release copies. [`MANIFEST.json`](MANIFEST.json) records hashes of the repository files. Stored outputs, private locations, Colab secret access, automatic model publication, and participant-record displays were removed or parameterized. Training settings and scientific scoring functions were retained.

Raw and participant-level derived records, CVs, model weights, and private study repositories are outside this code release. The code is distributed under the [MIT License](LICENSE). This license does not grant access to excluded study records or model weights, or replace any third-party model or dependency terms.

## Citation

See [`CITATION.cff`](CITATION.cff) for the code authors and associated manuscript title. No journal publication, acceptance, or DOI is claimed by this repository.
