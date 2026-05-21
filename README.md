# MirrorQuant

MirrorQuant is an AI-first fintech platform that helps users discover stocks with the same hidden behavioral "DNA" as a reference stock during its strongest breakout period.

Instead of screening by sector, valuation, or static ratios, MirrorQuant learns discrete latent market regimes from historical market behavior using a Vector Quantized Variational Autoencoder (VQ-VAE). A user selects a hero stock and a historical window, such as `MSFT` in early 2023, and MirrorQuant extracts the latent profile behind that move. The platform then scans the current market for stocks whose learned regime vectors match the hero profile, even across unrelated sectors.

The result is a new class of "Mirror" stocks that conventional screeners are likely to miss.

## Demo Quick Start

If your goal is a pitch, class presentation, or portfolio demo, start with the lightweight mock app in [`mirrorquant_demo/`](/abs/path/c:/Users/cheng/Hand-drawn-agent/mirrorquant_demo).

This demo version is intentionally simple:

- Static mock data for hero stocks, mirror matches, market weather, and industry-chain links
- A small FastAPI app that serves JSON endpoints
- A polished single-page dashboard for storytelling
- No real VQ-VAE training required

### Demo Features

The demo currently includes:

- 3 hero stocks: `MSFT`, `NVDA`, `LLY`
- 3 matching modes: `Price DNA`, `Economic DNA`, `Social DNA`
- A Market Watch panel
- An Industry Chain explanation panel
- Mock similarity scores and regime labels

### Demo Folder Layout

```text
mirrorquant_demo/
  app.py
  data/
    heroes.json
    mirror_matches.json
    market_watch.json
    industry_chain.json
  static/
    index.html
    styles.css
    app.js
```

### How To Run The Demo

1. Create or activate a Python environment.
2. Install the existing repo requirements:

```bash
pip install -r requirements.txt
```

3. Start the demo API:

```bash
python -m uvicorn mirrorquant_demo.app:app --reload
```

4. Open:

```text
http://127.0.0.1:8000
```

### Demo API Endpoints

- `GET /api/heroes`
- `GET /api/mirrors?ticker=MSFT&mode=price_dna`
- `GET /api/market-watch`
- `GET /api/industry-chain/MSFT`
- `GET /health`

### Best Demo Flow

Use this sequence during a presentation:

1. Select `MSFT` as the hero stock.
2. Show the `Price DNA` matches first.
3. Toggle to `Economic DNA` and explain that the ranking changes because the macro backdrop matters.
4. Toggle to `Social DNA` and explain narrative similarity.
5. Use the Market Watch panel to frame today's regime.
6. End with the Industry Chain panel to show how AI-discovered relationships connect back to market logic.

### What To Say During The Demo

You can describe it like this:

> MirrorQuant is not screening for stocks in the same sector. It is retrieving stocks expressing the same latent breakout behavior under similar market conditions.

And if someone asks whether the model is live:

> This demo uses precomputed regime outputs so we can focus on product interaction and explainability. The production version would replace these mocks with a trained latent-regime pipeline.

## Why MirrorQuant

Traditional stock discovery tools are good at filtering for known characteristics. They are much weaker at answering questions like:

- Which stocks are behaving today like Microsoft did before its strongest breakout?
- Which names share the same accumulation, volatility compression, and macro backdrop even if they are in different industries?
- Which AI-discovered relationships make sense when viewed through real supply chains and sector dependencies?

MirrorQuant is designed to answer those questions with a combination of:

- Latent regime learning from market time series
- Multi-modal representation learning across price, macro, and sentiment
- A market watch layer for regime awareness
- An industry-chain knowledge layer for explainability and validation
- A Regime-as-a-Service API for downstream fintech products

## Core Product Experience

Users interact with MirrorQuant through a dashboard:

1. Select a hero stock and the time window representing its most successful breakout period.
2. Choose a matching mode:
   - `Price DNA`
   - `Economic DNA`
   - `Social DNA`
   - `Blended DNA`
3. Generate the hero embedding from the selected period.
4. Scan the live market for similar latent regime vectors.
5. Review ranked mirror candidates, regime labels, and supporting explanations.
6. Inspect the current market weather before acting on results.

## Product Modules

### 1. Mirror Discovery Engine

The core engine learns a compressed representation of stock behavior and retrieves stocks with similar latent states.

Inputs:

- OHLCV time-series data
- Rolling returns and volatility features
- Volume shock and liquidity proxies
- Relative strength and momentum features

Outputs:

- Regime code sequence
- Continuous latent embedding
- Similarity score between hero and candidate windows
- Regime confidence and transition markers

### 2. Multi-Modal Representation Layer

MirrorQuant extends beyond price action by combining:

- Market data: price, volume, realized volatility, relative strength
- Macroeconomic context: rates, inflation, yield curve, VIX, liquidity proxies
- Market sentiment: FinBERT-based news and sentiment embeddings

This enables multiple matching modes:

- `Price DNA`: pattern similarity from market microstructure and trend behavior
- `Economic DNA`: similarity in macro backdrop and regime conditions
- `Social DNA`: similarity in narrative, sentiment, and news tone

### 3. Market Watch Module

This module gives users a near-real-time view of current market conditions.

It should surface:

- Key macro indicators
- Volatility metrics
- Active regime states
- Regime transitions
- Alerts when the market enters or exits historically significant conditions

Purpose:

- Help users understand current market weather
- Prevent blind use of similarity results
- Explain whether a historical hero setup is compatible with today's regime

### 4. Industry Chain Module

The Industry Chain Module overlays a structured map of how companies connect across supply chains and sector exposures.

It is used to:

- Explain why two stocks may share similar latent behavior
- Validate AI-discovered relationships against real-world dependencies
- Show how quantitative signals propagate through upstream and downstream networks
- Bridge learned signals with fundamental market logic

### 5. Regime-as-a-Service API

MirrorQuant exposes learned regime signals through an API so other fintech products can consume them directly.

Potential consumers:

- Robo-advisors
- Quant dashboards
- Research terminals
- Portfolio monitoring systems
- Strategy orchestration layers

## Example User Journey

A user selects `Microsoft` from `2023-01-01` to `2023-04-01`.

MirrorQuant:

1. Encodes the period into a latent regime profile.
2. Filters out short-term noise through the learned representation.
3. Compares that profile with the latest market-wide embeddings.
4. Returns stocks currently expressing similar breakout behavior.
5. Explains whether the similarity is driven by price action, macro context, sentiment, or industry linkages.

The output is not "stocks in software" or "stocks with similar PE ratios." It is "stocks behaving today like the chosen hero stock behaved during its strongest move."

## High-Level Architecture

```text
+---------------------+       +---------------------+       +----------------------+
|   Web Dashboard     | <---> |    FastAPI Backend  | <---> |   Postgres/pgvector  |
| React / Next.js     |       | query + orchestration|      | app data + embeddings|
+---------------------+       +---------------------+       +----------------------+
                                        |
                                        v
                              +----------------------+
                              |   ML Pipeline        |
                              | PyTorch + VQ-VAE     |
                              | feature engineering  |
                              +----------------------+
                                        |
                                        v
                              +----------------------+
                              |  Data Sources        |
                              | OHLCV / Macro / News |
                              +----------------------+
                                        |
                                        v
                              +----------------------+
                              | Industry Chain Layer |
                              | graph-style metadata |
                              +----------------------+
```

## Technical Design

### Frontend

- Interactive dashboard for hero selection, mirror search, and market watch
- Charting for hero window, candidate comparisons, and regime transitions
- Mode toggles for `Price DNA`, `Economic DNA`, `Social DNA`, and blended search

Suggested stack:

- `Next.js` or `React + Vite`
- `TypeScript`
- `Tailwind CSS`
- `ECharts`, `Plotly`, or `Recharts`

### Backend API

The backend coordinates data retrieval, embedding generation, similarity search, and dashboard responses.

Suggested stack:

- `FastAPI`
- `Pydantic`
- `PostgreSQL`
- `pgvector`
- `Redis` for caching or jobs when needed

Suggested endpoints:

- `POST /hero/encode`
- `GET /mirrors`
- `GET /market-watch`
- `GET /regimes/current`
- `GET /industry-chain/{ticker}`
- `POST /api/regimes/query`

### ML Pipeline

The ML layer is the heart of MirrorQuant.

Suggested components:

- Feature engineering for rolling time-series windows
- Sequence encoder for price and volume features
- VQ-VAE for discrete latent regime learning
- Similarity retrieval over current-market embeddings
- Offline evaluation against historical breakout analogs

Suggested stack:

- `PyTorch`
- `pandas`
- `numpy`
- `scikit-learn`
- `faiss` or `pgvector` for nearest-neighbor search

### Data Layer

MirrorQuant should begin with structured, versioned datasets.

Core categories:

- Daily OHLCV market data
- Macro indicator snapshots
- News and sentiment embeddings
- Company metadata
- Industry chain relationships

## Data Model

This is a practical starting schema.

### Stocks

```text
stocks(
  id,
  ticker,
  name,
  sector,
  industry,
  exchange,
  market_cap_bucket
)
```

### Historical Windows

```text
windows(
  id,
  stock_id,
  start_date,
  end_date,
  feature_version,
  label
)
```

### Regime Embeddings

```text
embeddings(
  id,
  window_id,
  mode,
  vector,
  regime_code,
  confidence,
  created_at
)
```

### Mirror Matches

```text
mirror_matches(
  id,
  hero_window_id,
  candidate_stock_id,
  candidate_window_id,
  mode,
  score,
  rationale,
  created_at
)
```

### Macro Snapshots

```text
macro_snapshots(
  date,
  policy_rate,
  inflation,
  vix,
  yield_curve_spread,
  liquidity_proxy,
  regime_state
)
```

### News Features

```text
news_features(
  id,
  stock_id,
  date,
  sentiment_score,
  finbert_embedding,
  article_count
)
```

### Industry Chain Edges

```text
industry_edges(
  id,
  source_stock_id,
  target_stock_id,
  relation_type,
  weight,
  notes
)
```

## Suggested Development Phases

### Phase 1: MVP

Goal: prove that latent similarity retrieval is useful.

Scope:

- 100 to 500 stocks
- Daily OHLCV data only
- One matching mode: `Price DNA`
- Hero window selection
- Similarity retrieval for current candidates
- Basic dashboard and API

Deliverables:

- First embedding pipeline
- First mirror search endpoint
- First interactive dashboard

### Phase 2: Multi-Modal Expansion

Goal: make matching more robust and explainable.

Add:

- Macro features
- Sentiment embeddings
- Mode toggles
- Market Watch Module

### Phase 3: Explainability and Platformization

Goal: connect latent quant signals to market logic and external products.

Add:

- Industry Chain Module
- Regime transition explanations
- Regime-as-a-Service API
- Portfolio-level downstream use cases

## MVP Success Criteria

The MVP should answer these questions well:

- Can the system retrieve plausible analog stocks from current market data?
- Do the retrieved matches outperform naive sector or factor screens in relevance?
- Can a user understand why a match was returned?
- Can the dashboard present market regime context clearly enough to support decisions?

## Recommended Project Structure

```text
mirrorquant/
  frontend/
  backend/
    app/
      api/
      services/
      schemas/
      models/
  ml/
    data/
    features/
    training/
    inference/
    evaluation/
  data/
    samples/
    industry_chain/
  docs/
```

## What To Build First

If you are starting from scratch, build in this order:

1. Define the stock universe and collect historical OHLCV data.
2. Create rolling windows and engineered price-volume features.
3. Train a baseline encoder, then upgrade to a VQ-VAE.
4. Store hero and current-market embeddings.
5. Build similarity search and ranking.
6. Expose results through a FastAPI endpoint.
7. Add a simple dashboard for search and visualization.
8. Add macro, sentiment, and industry-chain layers after the core retrieval works.

## Future Extensions

- Intraday regime detection
- Portfolio-level mirror search
- Regime-conditioned alerting
- Supply-chain shock propagation views
- Cross-asset regime matching
- Model monitoring and drift detection
- Personalized mirror watchlists

## Status

MirrorQuant is currently at the concept and architecture stage. The immediate next step is to build the MVP around `Price DNA` and validate whether latent-regime retrieval produces useful analog matches in live market conditions.

For demos, a working mock experience now exists in [`mirrorquant_demo/app.py`](/abs/path/c:/Users/cheng/Hand-drawn-agent/mirrorquant_demo/app.py:1) with supporting data under [`mirrorquant_demo/data/`](/abs/path/c:/Users/cheng/Hand-drawn-agent/mirrorquant_demo/data).

## License

Add your preferred license here.


## To start 
python -m uvicorn mirrorquant_demo.app:app --reload