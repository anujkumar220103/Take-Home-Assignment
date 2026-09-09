# TWCS Dataset Understanding and Intent Baseline

This project implements Step 1 and a deliberately limited Step 2 baseline for the
AmazonHelp customer-support assignment. Step 1 validates and prepares the data.
Step 2 discovers a transparent first-pass taxonomy and trains an interpretable text
classifier. It does not implement Step 3 retrieval or reply generation.

## Input

Place the original dataset at:

```text
data/raw/twcs.csv
```

The pipeline expects the standard TWCS columns:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The validator accepts harmless extra columns, but it does not silently rename or
invent missing required fields.

## Run the complete Step 1 pipeline

From the project root:

```powershell
python -m src.data.run_step1
```

Optional arguments:

```powershell
python -m src.data.run_step1 --input data/raw/twcs.csv --sample-size 10000 --seed 20260910
```

The command writes JSON reports under `reports/`, processed rows under
`data/processed/`, and the reproducible development sample under `data/samples/`.

## Scope boundary

Step 1 intentionally does not implement LLM calls, embeddings, vector databases,
RAG, reply generation, escalation models, golden sets, LLM judges, or final
evaluation metrics. Step 2 adds only the intent baseline described below.

## Step 2: Intent Discovery and Classification

### What it does

Step 2 identifies recurring customer-support themes in AmazonHelp inbound messages,
assigns transparent silver labels for a first benchmark, trains a text classifier,
and evaluates it on conversation groups not used for training.

### Taxonomy derived from AmazonHelp data

The taxonomy was derived by reviewing recurring vocabulary in the real AmazonHelp
customer sample and writing small, inspectable keyword rules. Ambiguous or
multi-theme messages become `other_or_unclear`; these labels are not human ground
truth.

The current final taxonomy has **9 intents**:

- `delivery_tracking`: package location, tracking, shipment, or delivery status.
- `delivery_delay`: late, missed, overdue, or promised-date delivery problems.
- `order_changes_cancellation`: changing, correcting, or cancelling an order.
- `returns_refunds`: returns, refunds, reimbursements, or money-back requests.
- `missing_wrong_damaged_item`: missing, wrong, damaged, broken, or defective items.
- `account_login`: sign-in, password, account access, or account security.
- `payment_billing`: charges, billing, cards, gift cards, or payment failures.
- `prime_subscription`: Prime membership, trials, renewals, or subscription benefits.
- `other_or_unclear`: no single clear match from the current taxonomy.

### Classification and validation

The baseline uses word-level TF-IDF unigrams and bigrams with balanced logistic
regression. The split is deterministic, uses seed `20260910`, and groups rows by
their parent tweet ID when available. Tweet IDs are excluded from features. Labels
are created before the split by transparent rules, so the reported metrics measure
agreement with held-out silver labels, not human-annotated accuracy.

Run Step 2 from the project root:

```powershell
python -m src.data.run_step2
```

Outputs are written to `reports/step2_results.json`,
`reports/step2_intent_report.md`, and
`data/processed/step2_customer_intents.csv` (the generated CSV is ignored by Git).
The report contains frequencies, representative real examples, boundaries,
confusion matrix, error analysis, and low-confidence cases.

The current validation result is documented in the report and is explicitly not a
production claim. On the current 2,682-row conversation-group validation split,
the baseline achieved **0.8941 accuracy**, **0.6829 macro precision**,
**0.8614 macro recall**, and **0.7552 macro F1** against silver labels. There
were **646 low-confidence predictions (24.09%)** below the 0.45 confidence
threshold. The report remains the authoritative source for the confusion matrix
and error details.

Known limitations include silver rather than human labels, severe class imbalance,
multilingual and ambiguous messages, sparse minority intents, and a text-only
classifier. A separately annotated evaluation set is required before claiming
production-level accuracy.
