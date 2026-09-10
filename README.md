# AmazonHelp AI Support Agent

This project implements Steps 1-4 of the assignment in a deliberately simple, auditable way.

- Step 1 validates and prepares the TWCS dataset.
- Step 2 discovers a transparent intent taxonomy and trains a baseline classifier.
- Step 3 retrieves historical AmazonHelp cases and generates grounded replies.
- Step 4 prepares a human-labelled golden set and provides evaluation utilities.

Step 4 does not fabricate human labels or judge results. The golden set is frozen and manually labelled; the evaluator records unavailable LLM-judge results when no API key is configured.

## Results

The latest frozen evaluation uses 200 golden examples. Main-agent intent accuracy is `0.8050` and macro F1 is `0.7969`. The trivial majority baseline has accuracy `0.1300` and macro F1 `0.0256`; the simple TF-IDF baseline has accuracy `0.8050` and macro F1 `0.7969`. Human-rated mean reply quality is `0.6750`, mean grounding quality is `0.6788`, false auto-handle rate is `0.6835`, and false escalation rate is `0.2143`.

LLM judge coverage is `0/200` in the current environment because `OPENAI_API_KEY` is not configured. Human/LLM agreement is therefore unavailable, not fabricated. See [reports/final_report.md](reports/final_report.md), [reports/step4_evaluation_report.md](reports/step4_evaluation_report.md), and [reports/step4_failure_analysis.md](reports/step4_failure_analysis.md).

## Repository Structure

```text
data/raw/             local input dataset, ignored by Git
data/golden/          annotation-ready golden set and manifest
reports/              Step 1-4 reports and annotation guidelines
src/data/             Step 1 and Step 2 pipelines
src/agent/            Step 3 retrieval and reply agent
src/evaluation/       Step 4 preparation and evaluation utilities
tests/                offline behavior tests
```

## Setup and Installation

Use Python 3.10 or newer. Install the project dependency set:

```powershell
python -m pip install -r requirements.txt
```

No API key is needed for Steps 1, 2, 4 preparation, tests, or the Step 3 mock demo.

## Input

Place the original dataset at:

```text
data/raw/twcs.csv
```

Required columns:

- `tweet_id`: tweet identifier
- `author_id`: tweet author or support handle
- `inbound`: whether the message is inbound to the support account
- `created_at`: tweet timestamp
- `text`: original message text
- `response_tweet_id`: child response ID or IDs
- `in_response_to_tweet_id`: parent tweet ID for a reply

## Step 1

Run the complete data pipeline:

```powershell
python -m src.data.run_step1
```

It validates the raw data, profiles quality, analyzes brands and conversations, selects AmazonHelp, creates processed data, and creates a deterministic conversation-centered sample.

Outputs are written under `reports/`, `data/processed/`, and `data/samples/`. Generated CSVs and the raw CSV are ignored by Git.

For a quick reproduction using existing derived artifacts, Step 1 does not need to be rerun before Steps 2-4. Running it against the full 2.8M-row dataset is the slower path; the generated sample already exists for lightweight development.

## Step 2

Run intent discovery and classification:

```powershell
python -m src.data.run_step2
```

The current taxonomy has nine transparent silver-label intents: delivery tracking, delivery delay, order changes/cancellation, returns/refunds, missing/wrong/damaged item, account/login, payment/billing, Prime/subscription, and other/unclear.

The classifier uses TF-IDF word unigrams/bigrams with balanced logistic regression. It uses a conversation-group split and seed `20260910`. Current metrics are silver-label agreement only: accuracy `0.8941`, macro precision `0.6829`, macro recall `0.8614`, and macro F1 `0.7552`. These are not production claims.

## Step 3

Run the retrieval-grounded agent demo:

```powershell
python -m src.agent.run_demo "Where is my package? It is late."
```

The default retrieval corpus is the full raw TWCS file. It contains **168,814** observed customer-to-AmazonHelp interactions. Retrieval uses TF-IDF similarity reranked with intent compatibility, response availability, and conversation quality.

The deterministic fallback uses an intent-specific template and never claims private account access or copies a historical response verbatim. Account-specific requests escalate. Optional LLM mode requires `OPENAI_API_KEY` and uses `OPENAI_BASE_URL` and `OPENAI_MODEL` when provided; network/API errors fall back to mock mode.

The provisional thresholds are retrieval evidence `0.08` and intent confidence `0.45`. Step 4 must tune and evaluate them; they are not claimed to be optimal.

## Step 4: Golden Set and Evaluation

### Prepare the golden set

```powershell
python -m src.evaluation.prepare_golden --size 200 --seed 20260910
```

This creates:

- `data/golden/golden_set.csv`
- `data/golden/golden_manifest.json`

The sample is drawn from real linked customer/AmazonHelp interactions. Sampling is deterministic and spread across the existing silver intent groups for coverage only. The `silver_sampling_group` column is not a human label.

### Manual labeling

Read [reports/golden_annotation_guidelines.md](reports/golden_annotation_guidelines.md). For every row, manually fill:

Start the resumable one-example-at-a-time annotation tool with:

```powershell
python -m src.evaluation.annotate_golden
```

The tool displays only the example ID, customer message, and available prior support response. It intentionally hides silver groups, model predictions, retrieved evidence, model responses, and model decisions. It saves after every completed row and resumes at the first incomplete row. Before the first write it creates `data/golden/golden_set.backup.csv` once; this backup is ignored by Git.

- `human_intent`
- `human_escalation`
- `human_reply_quality`
- `human_grounding_quality`
- `annotator_notes`
- `annotation_status` as `complete`

Do not copy `silver_sampling_group` into `human_intent`. Do not replace missing labels with model predictions.

### Run evaluation

```powershell
python -m src.evaluation.run_evaluation
```

The current golden set has been manually labelled and passes schema validation. The evaluation command runs the frozen main agent, a majority-intent trivial baseline, a simple TF-IDF baseline, offline metrics, optional judge scoring, agreement where available, and failure analysis. Do not treat Step 2 silver metrics as golden-set results. The evaluator verifies that the golden file's SHA-256 is unchanged.

The evaluation writes `reports/evaluation_results.json`, `reports/step4_evaluation_report.md`, and `reports/step4_failure_analysis.md`. LLM judge fields remain unavailable when `OPENAI_API_KEY` is not configured.

## Step-by-Step Reproduction

From the project root:

```powershell
# Step 1 exploration and preparation, only if raw data is present
python -m src.data.run_step1

# Step 2 intent pipeline using the existing development sample
python -m src.data.run_step2

# Step 3 mock demo using the full retrieval corpus
python -m src.agent.run_demo "Where is my package? It is late."

# Step 4 annotation-ready preparation
python -m src.evaluation.prepare_golden --size 200 --seed 20260910

# Step 4 evaluation: runs offline metrics and optional LLM judging
python -m src.evaluation.run_evaluation

# Final checks
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```

The lightweight path uses existing Step 1 and Step 2 artifacts and normally completes within the assignment's 15-minute reproduction target on the supplied workspace. Full Step 1 and full-corpus Step 3 processing depend on local disk, memory, and the 2.8M-row raw dataset; their runtime is not represented as a guaranteed benchmark here.

## Demo Output

The Step 3 command prints JSON containing the input message, predicted intent and confidence, ranked historical cases, component scores, generated response, action, evidence strength, and decision reason. Exact scores can change if the local derived artifacts change.

## Tests

Run all offline behavior tests:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Tests do not call live APIs.

## Limitations

- Step 2 labels are silver labels, not human ground truth.
- Step 3 retrieval is lexical and may miss paraphrases or multilingual matches.
- Step 3 cannot inspect or change private customer accounts.
- Step 4 final metrics and failure analysis are written by the evaluation command. LLM judge results and human/judge agreement remain unavailable when `OPENAI_API_KEY` is not configured.
- No Step 5 functionality is included.

## Failure Modes

The five prioritized golden-set failure analyses are written to `reports/step4_failure_analysis.md` from actual evaluation predictions and human ratings. They are not inferred from Step 3 demo behavior.

## What I Would Do With One More Week

The conditional improvement plan is in `reports/step4_evaluation_report.md` and is tied to failures that must first be observed in the completed golden set.

## Decision Log

See [reports/decision_log.md](reports/decision_log.md) for the engineering choices made across Steps 1-4 and their trade-offs.
