# Architecture

## System overview

```mermaid
flowchart LR
    subgraph Offline["Offline: Training Pipeline"]
        A[UCI Bike Sharing\nGitHub mirror] -->|src/ingest.py| B[data/hour_raw.csv]
        B -->|src/features.py| C[Lag + rolling + cyclical features]
        C -->|src/train.py| D[Time-based split\nexpanding-window CV]
        D --> E[(model.joblib\npreprocessor.joblib)]
        D --> F[metrics.json\nvs. Ridge + seasonal-naive baselines]
        E --> G[Explainability\nsrc/explain.py]
        G --> H[SHAP summary plot]
    end

    subgraph Online["Online: Serving"]
        I[Client] -->|POST /predict| J[FastAPI\napi/main.py]
        J --> K[DemandModel\nsrc/predict.py]
        K -->|loads| E
        K --> J
        J -->|predicted_count| I
    end

    subgraph CI["CI/CD - GitHub Actions"]
        L[git push] --> M[Install deps]
        M --> N[Ingest + Train]
        N --> O[pytest + coverage]
        O --> P[Docker build]
    end
```

## Time-based validation (the part most tutorials get wrong)

```mermaid
flowchart TB
    A["Full dataset, sorted chronologically\n(2011-01-01 -> 2012-12-31)"] --> B["Time cutoff: last 120 days"]
    B --> C["Train set\n(everything before cutoff)"]
    B --> D["Test set\n(last 120 days only)"]
    C --> E["TimeSeriesSplit expanding-window CV\n(5 folds, each fold's validation\nis strictly after its training data)"]
    E --> F["Final model fit on full train set"]
    F --> G["Evaluate once on held-out test set"]
```

A **random** train/test split or plain K-fold CV on a time series lets the
model train on rows that happen to fall chronologically *after* some of its
validation rows — e.g. training on 3pm data from a Tuesday while "predicting"
9am from the same day. That leaks information no real deployment would ever
have, and inflates validation scores in a way that doesn't survive contact
with production. This project uses:

- A hard **chronological cutoff** for the final train/test split (test = the
  most recent 120 days, exactly as if this were deployed today and asked to
  forecast the next few months)
- **`TimeSeriesSplit`** (expanding-window CV) instead of `KFold` for
  hyperparameter/model validation during development

## Design decisions

| Decision | Rationale |
|---|---|
| Lag features (1h, 24h, 168h) + rolling means | The single strongest predictor of "bikes rented this hour" is "bikes rented recently" — lag features let a tree-based model exploit that directly instead of re-deriving it from raw calendar fields. |
| Cyclical (sin/cos) encoding of hour/month/weekday | Hour 23 and hour 0 are one hour apart in reality but far apart as raw integers; sin/cos encoding preserves that adjacency, which matters for late-night demand patterns. |
| `HistGradientBoostingRegressor` vs a linear baseline | Demand has non-linear interactions (e.g. weather matters much more on weekends than on commute-heavy weekdays); the results table shows the tree model beating a comparable linear (Ridge) model by a wide margin, which justifies the extra complexity. |
| Explicit seasonal-naive baseline ("same hour last week") | Any forecasting model needs a trivial baseline to beat, or its apparent skill is illusory. This project reports the naive baseline's error alongside the model's, and tests (`test_model_beats_seasonal_naive_baseline`) enforce that the trained model actually outperforms it. |
| Single `ColumnTransformer` persisted with the model | Same rationale as any production ML system — eliminates train/serve skew. |
| API takes pre-computed lag/rolling features rather than raw timestamps | This project scores a single hour given its recent history (a "nowcast"/point regression), not a full autoregressive multi-step forecast — clearly documented as a scope boundary, with the natural extension (recursive multi-step forecasting) noted in Future Work. |

## Data flow

1. `src/ingest.py` downloads the real UCI Bike Sharing hourly dataset from a GitHub mirror.
2. `src/features.py` builds lag features, rolling means, and cyclical encodings, dropping the earliest 168 rows that lack a full week of history.
3. `src/train.py` performs a chronological train/test split, runs expanding-window CV, trains `HistGradientBoostingRegressor`, and compares it against a Ridge regression baseline and a seasonal-naive baseline.
4. `src/explain.py` computes SHAP values on a sample of test-period hours.
5. `api/main.py` loads the persisted artifacts once at startup and serves `/predict` and `/health`.
