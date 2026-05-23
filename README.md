# MirrorQuant

MirrorQuant is an AI-first fintech platform that helps users discover stocks with the same hidden behavioral "DNA" as a reference stock during its strongest breakout period.

Instead of screening by sector, valuation, or static ratios, MirrorQuant learns discrete latent market regimes from historical market behavior using a Vector Quantized Variational Autoencoder (VQ-VAE). A user selects a hero stock and a historical window, such as `MSFT` in early 2023, and MirrorQuant extracts the latent profile behind that move. The platform then scans the current market for stocks whose learned regime vectors match the hero profile, even across unrelated sectors.

The result is a new class of "Mirror" stocks that conventional screeners are likely to miss.

## Current Demo Status

The repo now contains a working single-page demo in [`mirrorquant_demo/`](/abs/path/c:/Users/cheng/Desktop/Hand-drawn-agent/mirrorquant_demo) with:

- a FastAPI backend
- a dark quant-terminal-style dashboard
- a real `Price DNA` path powered by a trained VQ-VAE
- curated fallback demo data for `Economic DNA` and `Social DNA`
- user-selectable hero windows via start and end date inputs

The app mixes real and mock layers on purpose:

- `Price DNA` uses the trained VQ-VAE pipeline
- `Economic DNA` and `Social DNA` still use curated demo results from JSON
- `Market Watch` and `Industry Chain` remain demo data

## Demo Folder Layout

```text
mirrorquant_demo/
  app.py
  build_training_data.py
  train_vqvae.py
  encode_windows.py
  vqvae_search.py
  fetch_prices_alpaca.py
  fetch_prices_moomoo.py
  fetch_prices_alpha_vantage.py
  data/
    heroes.json
    mirror_matches.json
    market_watch.json
    industry_chain.json
    prices.csv
    training_windows.npz
    training_windows_meta.csv
    vqvae_model.pt
    window_embeddings.npz
    window_embedding_meta.csv
  static/
    index.html
    styles.css
    app.js
```

## Install

1. Create or activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

Start the FastAPI app:

```bash
python -m uvicorn mirrorquant_demo.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Main User Flow

1. Select a hero stock.
2. Choose a DNA mode.
3. Optionally edit the `Start date` and `End date`.
4. Click `Find Mirrors`.
5. Review:
   - selected hero window
   - encoded model window
   - top mirror matches
   - market watch context
   - industry-chain explanation

Important note:

- the **Selected window** is the date range the user asked for
- the **Encoded window** is the exact model-sized slice used by the VQ-VAE

Because the current VQ-VAE was trained on fixed 40-day windows, the app encodes a 40-trading-day slice from the chosen hero range.

## API Endpoints

- `GET /api/heroes`
- `GET /api/mirrors?ticker=MSFT&mode=price_dna`
- `GET /api/mirrors?ticker=MSFT&mode=price_dna&start_date=2023-01-03&end_date=2023-04-03`
- `GET /api/market-watch`
- `GET /api/industry-chain/MSFT`
- `GET /health`

## Fetch Real Market Data With Alpaca

The repo includes an Alpaca fetch script that replaces the demo `prices.csv` with real daily OHLCV stock bars:

[`mirrorquant_demo/fetch_prices_alpaca.py`](/abs/path/c:/Users/cheng/Desktop/Hand-drawn-agent/mirrorquant_demo/fetch_prices_alpaca.py:1)

Alpaca docs:

- Historical stock bars endpoint: https://docs.alpaca.markets/us/reference/stockbars
- Market Data getting started guide: https://docs.alpaca.markets/us/docs/getting-started-with-alpaca-market-data

### Step 1: Create an Alpaca account and API keys

According to Alpaca's Market Data getting started guide, you generate keys from the Alpaca dashboard after creating an account.

### Step 2: Set your Alpaca credentials

Add them to your environment or a local `.env` file:

```env
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
```

### Step 3: Install Alpaca support

If needed:

```bash
pip install alpaca-py
```

### Step 4: Fetch prices

Fetch the default demo tickers:

```bash
python mirrorquant_demo/fetch_prices_alpaca.py
```

Fetch a custom date range:

```bash
python mirrorquant_demo/fetch_prices_alpaca.py --start 2023-01-01 --end 2024-12-31
```

Fetch specific tickers:

```bash
python mirrorquant_demo/fetch_prices_alpaca.py --tickers MSFT NVDA LLY AAPL
```

The script defaults to the `iex` feed, which is the safest default for a basic setup. You can also request `sip` if your account has access:

```bash
python mirrorquant_demo/fetch_prices_alpaca.py --feed sip
```

## Optional moomoo Path

If you want to pull price history through your moomoo account instead of Alpaca, the repo now also includes:

[`mirrorquant_demo/fetch_prices_moomoo.py`](/abs/path/c:/Users/cheng/Desktop/Hand-drawn-agent/mirrorquant_demo/fetch_prices_moomoo.py:1)

This path uses moomoo OpenAPI, which requires two parts:

1. The Python SDK:

```bash
pip install moomoo-api
```

2. A locally running `OpenD` gateway signed in to your moomoo account.

Official docs:

- OpenAPI intro: https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html
- OpenD setup: https://openapi.moomoo.com/moomoo-api-doc/en/opend/opend-cmd.html
- Python sample / install notes: https://openapi.moomoo.com/moomoo-api-doc/en/quick/demo.html
- Historical candles API: https://openapi.moomoo.com/moomoo-api-doc/en/quote/request-history-kline.html

### Step 1: Add your local OpenD settings

Set these in your environment or local `.env` file:

```env
MOOMOO_HOST=127.0.0.1
MOOMOO_PORT=11111
MOOMOO_MARKET_PREFIX=US
```

### Step 2: Start OpenD and log in

Start `OpenD` on your machine and sign in with your moomoo account before running the script. The fetcher talks to OpenD on `127.0.0.1:11111` by default.

### Step 3: Fetch prices

Fetch the default demo tickers:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py
```

Fetch a custom date range:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py --start 2023-01-01 --end 2024-12-31
```

Fetch specific tickers:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py --tickers MSFT NVDA LLY AAPL
```

If you want to pass already-prefixed symbols, that works too:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py --tickers US.MSFT US.NVDA SG.D05
```

Adjustment mode defaults to `qfq`, and you can switch it if needed:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py --adjust none
```

Important note:

- moomoo applies historical candlestick quotas and request limits, so very large symbol/date pulls may need to be split up.

### Optional: Power The Dashboard With Live Market Watch Proxies

The dashboard can also use a separate proxy price file for real market-watch sparklines. Fetch ETF proxies into `market_watch_prices.csv` like this:

```bash
python mirrorquant_demo/fetch_prices_moomoo.py --tickers SPY QQQ IWM TLT HYG --start 2024-01-01 --output mirrorquant_demo/data/market_watch_prices.csv
```

If that file exists, the `Market Watch` panel will use those real time series instead of demo-generated sparklines.

## Optional Alpha Vantage Path

If you still want a second provider option, the repo also includes:

[`mirrorquant_demo/fetch_prices_alpha_vantage.py`](/abs/path/c:/Users/cheng/Desktop/Hand-drawn-agent/mirrorquant_demo/fetch_prices_alpha_vantage.py:1)

## Retrain The AI

If you change `prices.csv`, you should rebuild the VQ-VAE pipeline in this exact order.

### 1. Build training windows

This converts daily OHLCV history into fixed-size model windows:

```bash
python mirrorquant_demo/build_training_data.py
```

Outputs:

- `mirrorquant_demo/data/training_windows.npz`
- `mirrorquant_demo/data/training_windows_meta.csv`

### 2. Train the VQ-VAE

This trains the current `Price DNA` model:

```bash
python mirrorquant_demo/train_vqvae.py
```

Output:

- `mirrorquant_demo/data/vqvae_model.pt`

### 3. Encode all windows

This builds the searchable embedding library used by the app:

```bash
python -m mirrorquant_demo.encode_windows
```

Outputs:

- `mirrorquant_demo/data/window_embeddings.npz`
- `mirrorquant_demo/data/window_embedding_meta.csv`

### 4. Restart the app

After retraining and re-encoding, start or restart:

```bash
python -m uvicorn mirrorquant_demo.app:app --reload
```

## When You Need To Retrain

You do **not** retrain the AI every time a user clicks `Find Mirrors`.

You typically retrain only when:

- you replace demo price data with real price data
- you add more stocks
- you change the feature engineering
- you change the VQ-VAE architecture

You should re-encode windows when:

- you retrain the model
- you update `prices.csv`

## What Is Real vs Mock Right Now

### Real

- FastAPI backend
- custom hero window input
- VQ-VAE training pipeline
- VQ-VAE embedding search for `Price DNA`

### Mock / curated

- `Economic DNA`
- `Social DNA`
- market watch values
- industry chain relationships

## Best Demo Flow

Use this sequence during a presentation:

1. Start with `MSFT` in `Price DNA`.
2. Show the selected window and the encoded model window.
3. Explain that the VQ-VAE is matching latent behavior, not sector labels.
4. Switch to `Economic DNA` and `Social DNA` as explainability layers.
5. Use Market Watch to frame the current regime.
6. Use Industry Chain to connect quant signals back to real-world logic.

## What To Say During The Demo

You can describe it like this:

> MirrorQuant is not screening for stocks in the same sector. It is retrieving stocks expressing the same latent breakout behavior under similar market conditions.

If someone asks whether the AI is real:

> Price DNA is powered by a trained VQ-VAE over rolling price-volume windows. Economic DNA and Social DNA are currently curated demo layers while the full multi-modal stack is still being built.

## Technical Summary

### Frontend

- single-page dashboard
- custom hero window inputs
- sci-fi quant-terminal visual system

### Backend

- `FastAPI`
- JSON-backed demo metadata
- VQ-VAE search orchestration for `Price DNA`

### ML Pipeline

- rolling 40-day windows
- sequence features:
  - `daily_return`
  - `volume_change`
  - `price_rel`
- VQ-VAE training with PyTorch
- offline embedding generation

## Future Extensions

- real macroeconomic feeds for `Economic DNA`
- sentiment/news embeddings for `Social DNA`
- longer-history or variable-length sequence models
- richer vector search and ranking logic
- realtime refresh jobs
- portfolio-level analog search

## License

Add your preferred license here.
