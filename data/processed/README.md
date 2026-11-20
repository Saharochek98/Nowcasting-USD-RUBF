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

### Особенности временных меток (`end`)

В исходном датасете MOEX ISS API встречаются различные значения времени окончания дневных свечей:

- `23:59:59` — 1083 наблюдения;
- `23:59:41/42` — 42 наблюдения (2022 год);
- `19:30:21/22` — 35 наблюдений (2024–2025 годы).

Для всех 35 свечей с окончанием около 19:30 проведено сравнение с минутными данными MOEX. Результаты согласуются с формированием свечей по сделкам текущего календарного дня до указанной временной границы. При этом точная техническая причина различий не установлена.

**Данные сохранены в исходном виде без корректировок**, поскольку они соответствуют значениям, опубликованным MOEX ISS API. Поле `end` не следует интерпретировать как точное время последней сделки.

### Торги в выходные дни

Датасет содержит **4 наблюдения за субботу**: `2024-04-27`, `2024-11-02`, `2024-12-28` и `2025-11-01`. Эти даты соответствуют официальным рабочим субботам Московской биржи. Отдельных наблюдений за воскресенья нет.

С 18 июля 2026 года валютные фьючерсы также торгуются в дополнительные сессии по выходным. Сделки таких сессий относятся к следующему торговому дню, поэтому могут учитываться в дневной свече понедельника без формирования отдельных строк за субботу и воскресенье.

**Все торговые даты сохранены в исходном виде**, включая рабочие субботы.
