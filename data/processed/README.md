# USDRUBF baseline dataset

This folder contains derived data intended for initial benchmarks. The dataset in `USDRUBF_baseline_2026-11-18.csv` has **1,160** daily observations from **2022-04-26** through **2026-11-18**, inclusive.

- `date`: trading-day label from the MOEX daily candle `begin` field.
- `price`: original MOEX daily candle `close` (RUB per USD); **not** the clearing settlement price.
- `volume`: daily candle trade volume (contracts).
- `price_type`: `CANDLE_CLOSE`, a label for the original field's semantics.
- `source`: MOEX ISS endpoint, futures / forts / rfud / USDRUBF / `interval=24`.

Source endpoint: https://iss.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/USDRUBF/candles.json?interval=24

**Provenance / limitations:** Derived from a full MOEX ISS daily candle export covering 2022-04-26–2026-11-18; `price` and `volume` are unadjusted, with no funding corrections or forward-fill. Preliminary checks found no null or nonpositive prices, duplicate dates, or OHLC consistency failures. A complete independent trading-calendar and historical candle availability audit is still pending. MOEX altered session-day attribution on 2026-03-23; `date` is not an exact execution or data-publication timestamp. Until actual availability is verified, follow the **conditional EOD / pseudo-PIT** convention in `docs/backtest_protocol.md`.

Keep the original 8-column OHLCV export separately under `data/raw/` when it is published. This simplified file does not replace the raw data.
