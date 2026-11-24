# Data handoff — USDRUBF (B1–B5)

**Дата:** 2026-11-23. **Статус:** набор данных готов для **условных EOD / pseudo-PIT** validation-экспериментов; **не** для утверждений о проверенном real-time PIT. Исследования B1–B4 завершены; B5 — передача без повторного аудита.

## Что использовать

| Объект | Файл |
|---|---|
| Модельная панель, 1160 дат (2022-04-26–2026-11-18) | `data/processed/features_daily_v1.csv` |
| Происхождение полей и пропусков | `data/processed/features_daily_v1_metadata.csv` |
| Состав сценариев | `configs/feature_scenarios_v1.json` |
| Загрузчик и инструкция | `scripts/build_model_features.py::load_scenario`, [features_daily_v1.md](features_daily_v1.md) |
| Правила оценивания | [backtest_protocol.md](backtest_protocol.md) |

**Сценарии:** `core` (**9** признаков, default); `core_ruonia_brent` (**11**) и `core_ruonia_brent_usdcnh` (**12**) — только отдельные проверки чувствительности к дополнительным допущениям. Не загружать все 12 полей как `core` автоматически. `swaprate_current` исключён.

```python
from scripts.build_model_features import load_scenario
rows = load_scenario(scenario="core")  # date + строго 9 разрешённых полей
```

## Что обязательно учитывать B/C/D

- **Target:** дневной `CANDLE_CLOSE` USDRUBF, `h=1..7` **следующих торговых сессий**, origin после условной доступности полной свечи. Не курс ЦБ и не settlement.
- **Funding:** у `swaprate_prev_session` **154 начальных пропуска**, первое значение **2022-12-06**; **198** наблюдаемых нулей не равны пропускам. Обработка missing, scaler, PCA и отбор признаков — **train-only** в каждом walk-forward окне; S0/S1/S2 описаны в B4 и требуют общего набора `(origin,target,h)` при сравнении.
- **PIT:** исторические `available_at` свечей и внешних значений не доказаны; CBR, RUONIA, Brent lag10, USD/CNH и funding имеют разные условные правила. Не повышать статусы до `verified`.
- **Календарь:** внутренний аудит пройден, **PROVISIONAL** до независимого подтверждения полного архива сессий.
- **Final test 2026:** не использовать для подбора факторов, гиперпараметров, импутации, интервалов или моделей.

## Проверка и подробности

```powershell
python scripts/build_model_features.py --compare
python scripts/audit_funding_missingness.py --compare
python -m unittest discover -s tests -p "test_*.py" -v
```

Подробные результаты **не дублируются**: [календарь](../reports/calendar_quality_report.md), [B1 — forecast origin](forecast_origin_pit_proposal.md), [B2 — PIT-каталог](feature_availability_catalog.csv), [B3 — coverage](../reports/features_daily_v1_coverage.md), [B4 — funding](../reports/funding_missingness_audit.md). Локально для B4 сообщено: **61 unittest (60 successful, 1 skipped — full external-source rebuild)**; повторный тест после B5 отдельно не заявлен.

**Граница завершения A (Data Engineer):** инженерные данные и ограничения переданы. Историческая PIT-сертификация, архивные источники и независимый календарь остаются открытыми исследованиями; B5 не меняет данные и бэктест.

**Роли для handoff:** A (Data Engineer) передаёт подготовленные B1–B5 артефакты участникам **B (Research Lead), C (Models), D (Uncertainty)**. «B1–B5» — исторические имена data-этапов, не роли людей.

**Роли команды:** A — Data Engineer, B — Research Lead, C — Models, D — Uncertainty. B1–B5 — названия этапов подготовки данных, а не роли участников.
