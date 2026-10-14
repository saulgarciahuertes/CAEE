# Reproducibility Scope And Known Limitations

## What Is Included

The recovered preparation/augmentation implementation, four ORPO training implementations, baseline/fine-tuned answer generation, direct GPT ratings, semantic-similarity scoring, generic prompt templates, extracted settings, and the dependencies declared by the original notebooks. The unfinished Llama judge experiment and unrelated sentiment/binary-classification experiments are excluded because they do not support the manuscript's reported comparison.

No raw survey records, participant-level derived datasets, generated answers, scored records, CVs, private dataset/model addresses, secrets, or weights are included. Code availability is not a claim that the full study can be reproduced from public inputs alone.

## Recovered Preparation Source

`notebooks/preparation/Redact_CVs_Augment_Survey_Data_and_Build_ORPO_Dataset.ipynb` retains preparation cells 4-9 from `emnlp.ipynb`. It excludes original authentication/setup, prototype Hugging Face conversion/publication, training, and older two-model evaluation. All 141 saved output objects and original metadata are excluded from this source-only release; the retained preparation cells themselves had no saved outputs. Cleared execution counts and incomplete saved history do not establish a complete historical execution sequence or every API payload.

The written flow uses Presidio configured for English `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, and `URL` detection on CV text, followed by case-sensitive literal name-part replacement. Survey identifier columns remain in the private master file but are excluded from the narrative field slices. Selected survey values may still contain identifying information. The instruction to omit institution/company names constrains generated output, not the request. Code and schema guards are not an anonymity certificate or evidence of institutional processing permission.

Narrative generation uses `gpt-3.5-turbo-0125`, three variants for each of six groups, and a 500-token output limit. Exact generic instructions are in `prompts/preparation_prompts.json`. Three CV variants x nine contextual variants x three answer variants x two fixed tasks yield 162 instruction rows per respondent. This accounts for 16,362 rows for the 101-record cohort, not 16,362 independent respondents. Locally comparing the private saved artifacts reproduced this entire prompt/question/answer multiset without new generation; no participant values are included in the repository.

Contrasting answers use the historical `gpt-3.5-turbo` alias, temperature 0.8, and a 500-token limit. Its resolved historical snapshot is not recorded. The Yes-to-No helper is retained but unused, and does not implement the reverse replacement.

The two later cleanup alternatives independently reread `orpo_data.csv`. Alternative B does not clean alternative A's output. Both use regex name replacement after generation in the written flow. Accent-normalization order, case sensitivity, placeholder spelling, and manual message-string serialization are preserved. In particular, `<PHONE NUMBER>` differs from Presidio's `<PHONE_NUMBER>`, and the source's unescaped message strings are not guaranteed valid JSON. The resulting CSVs are archival intermediates, not certified reusable training inputs. No missing split or conversion routine has been invented.

The release changes paths to environment-configured private locations and removes record/entity/CV diagnostics. It adds schema checks and explicit API/input-review opt-ins, which are release-only safeguards. The unchanged core functions and prompts are covered by offline synthetic checks. The selected historical dependency declarations do not specify a compatible English NLP model or a complete environment lock; Presidio model/resource compatibility and full pipeline execution have not been tested during release preparation.

## Data Partitions And Selection

Historical saved progress records corroborate upstream partition sizes of 14,725 training and 1,637 test rows, approximately 90/10. The command that created those partitions, its randomization method/seed, and respondent-disjointness were not recovered. This repository consumes the partitions; it does not recreate them or establish independent respondent-level validation.

Each training notebook shuffles the existing training partition with seed 42, selects its named row count, and reserves a further 1% for internal validation. The internal split has no explicit seed. The named counts are pre-validation selection sizes, not exact rows passed to the optimizer.

| Selected rows | Expected optimizer-training rows | Internal validation rows |
|---|---|---|
| 4096 | 4055 | 41 |
| 5120 | 5068 | 52 |
| 6144 | 6082 | 62 |
| 7168 | 7096 | 72 |

Inference selects the first 1,024 rows of the existing test partition. Both reported scoring notebooks sample 256 comparison rows with random state 42. They select the same records when the input CSV contents and order match. The private study exports were checked to agree on that sample, but are not included here.

## Models And Environment

The training and untuned answer-generation checkpoint is `meta-llama/Meta-Llama-3-8B-Instruct`. The GPT judge identifier is `gpt-3.5-turbo-0125`. The active SemScore encoder is `sentence-transformers/all-mpnet-base-v2`.

Fine-tuned model locations must be supplied by the caller. No default identifier for a private study model is embedded. Training copies save adapters and merged models locally instead of automatically uploading them. External experiment reporting is disabled.

Dependency files reproduce installation declarations, not a full environment lock. Some imports also relied on packages already available in the historical Colab environment. Package/API/model availability and GPU/runtime compatibility have not been freshly validated. The source uses historical interfaces, including `ORPOTrainer`, `ORPOConfig`, and `setup_chat_format`, that may differ across library releases. Do not infer a complete historical environment from an individual saved model's version metadata.

## Retained GPT Scoring Behavior

The source implements a direct-rating adaptation inspired by G-Eval. It contains no explicit chain-of-thought evaluation-step generation or rating-token probability weighting. Four integer ratings are requested; the code attempts parsing up to three times for each of five runs.

The original parser is case/order/spacing-sensitive and does not range-validate parsed integers. Each successful rating contributes its value divided by five. An unparsed run contributes zero under that fixed denominator. The release retains those functions for auditability; it does not silently redefine the reported experiment. Raw per-run judge responses and parse-success counters were not recovered, so individual-run success cannot be certified from valid-looking final averages alone.

Local checks of the private saved scores reproduced the manuscript's aggregate table by arithmetic averaging, without new judge or embedding-model calls. Such agreement verifies saved results and aggregation, not every historical scoring event. Five repeated runs do not by themselves establish consistency or quantify uncertainty.

## Retained Inference Behavior

The source uses its original plain-text role labels and answer-length/bracket instruction rather than a newly substituted chat template. It removes the input prompt from returned text and extracts the first curly-bracketed region, falling back to the whole answer if no pair is found. Generation settings, including the absence of an explicit seed or temperature override, are retained. Alternative prompt formatting or parsing would be a new methodological change, not simple release cleaning.

## Validation Performed

Notebook JSON and code-cell syntax were checked with shell installation lines treated as comments for Python parsing. Output/metadata removal, private-location parameterization, scientific settings/function retention, source-file hashes, repository-file hashes, credential patterns, and exclusion of study data/weights were checked. Isolated parsing functions are tested on synthetic strings and stubbed responses only.

No full training, inference, embedding evaluation, or external judge scoring was run during repository preparation. No new scientific result or metric was substituted for the supplied manuscript results.
