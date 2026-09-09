# TWCS AmazonHelp Support Agent

This project implements Steps 1-4 of the assignment in a deliberately simple, auditable way.

- Step 1 validates and prepares the TWCS dataset.
- Step 2 discovers a transparent intent taxonomy and trains a baseline classifier.
- Step 3 retrieves historical AmazonHelp cases and generates grounded replies.
- Step 4 prepares a human-labelled golden set and provides evaluation utilities.

Step 4 does not fabricate human labels or final metrics. The evaluation command refuses to run until the annotation file is completed.

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

The current repository stops at annotation-ready status because the golden set has not been manually labelled. After labeling, evaluation utilities support validation, confusion matrices, majority and silver-reference baselines, judge JSON parsing, and Cohen's kappa without live API calls.

The final report belongs at `reports/step4_evaluation_report.md`. It must contain results for intent, escalation, reply quality, LLM judge agreement, failure modes, and headline-number limitations only after those results actually exist.

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
- Step 4 final metrics, human agreement, LLM judge results, and failure analysis remain unavailable until real annotation is completed.
- No Step 5 functionality is included.
