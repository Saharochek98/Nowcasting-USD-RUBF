# features_daily_v1 — передача модельной панели участнику C

Версия схемы: `features_daily_v1`. Политика: `b2-approved-b3-v1`.
Основа: B1/B2, коммит `024e8d3`; ветка B3 — `feature/b-model-features-v1`.
Согласование списка и распределения факторов A явно сообщено пользователем
в задании B3. Старые документы B1/B2 сохраняют историческую отметку proposal;
конфигурация B3 фиксирует принятое для этой панели распределение сценариев.
Это не подтверждение новых исторических timestamps или строгого PIT.

## Назначение и файлы

Панель содержит **1160 строк**, 26.04.2022–18.11.2026. Режим —
**conditional EOD / pseudo-PIT**. Она является проекцией существующего снимка v2;
источники, лаги, календарь, цены и пропуски не пересчитываются и не заменяются.

| Файл | Назначение |
|---|---|
| `data/processed/model_dataset_usdrubf_all_features_v2.csv` | Неизменный вход, 1160 × 30 |
| `configs/feature_scenarios_v1.json` | Версия/источник/контрольная сумма, явные списки сценариев и запреты |
| `scripts/build_model_features.py` | Детерминированная проверка и проекция без обучения |
| `data/processed/features_daily_v1.csv` | Один общий store: date + объединение 12 экономических значений |
| `data/processed/features_daily_v1_metadata.csv` | Аудит: date + 23 служебных поля, без экономических значений текущего funding/value |
| `reports/features_daily_v1_coverage.md` | Детерминированный отчёт по сценариям, пропускам и SHA-256 |
| `tests/test_model_features.py` | Автономные проверки схемы, дат, проекции, сценариев и воспроизводимости |

Дополнительные проверяемые входы builder — исходные свечи, предварительный
торговый календарь и каталог B2. Пути заданы относительно корня репозитория,
не через локальные неотслеживаемые выгрузки. Внешние raw-export файлы для
проекции B3 не нужны. `.gitattributes` закрепляет LF для трёх генерируемых файлов,
чтобы Git checkout не нарушал побайтную воспроизводимость на Windows/Linux.

## Построение и сверка

Нужен Python 3.9+ и только стандартная библиотека. Из корня репозитория:

```powershell
python scripts/build_model_features.py
python scripts/build_model_features.py --compare
```

Первый запуск пишет оба CSV и coverage-report. `--compare` повторно строит их
в памяти, сравнивает все байты и ничего не записывает; несовпадение даёт ошибку.
Документ передачи C версионируется отдельно, не генерируется builder.
В выходных файлах нет текущего времени, случайных значений или SHA рабочего Git.
Исходная v2 закреплена SHA-256; другой снимок требует новой зарегистрированной версии.

Построение в отдельную папку для независимого сравнения:

```powershell
python scripts/build_model_features.py --output-dir results/features_v1_check --report results/features_v1_check/coverage.md
python scripts/build_model_features.py --output-dir results/features_v1_check --report results/features_v1_check/coverage.md --compare
```

Скрипт не перезаписывает входы и посторонние существующие артефакты репозитория.
Повторный запуск с default-путями заменяет только два собственных CSV и свой отчёт.
Вывод содержит SHA-256. Подробные контрольные суммы — в coverage-report.

## Предикторы и сценарии

Один store позволяет избежать трёх почти одинаковых CSV. Наличие значения в
store **не включает его в модель**: набор определяет конфигурация и загрузчик.
Не использовать `df.drop(columns='date')` для автоматического выбора всех
столбцов общего файла — так группа B попала бы в core без выбора сценария.

| Сценарий / ключ конфигурации | Признаков | Дополнение к core | Назначение |
|---|---:|---|---|
| A / `core` (по умолчанию) | 9 | — | Основной согласованный условный набор |
| B1 / `core_ruonia_brent` | 11 | RUONIA, Brent lag10 | Чувствительность к дополнительным PIT-допущениям |
| B2 / `core_ruonia_brent_usdcnh` | 12 | RUONIA, Brent lag10, USD/CNH previous date | Чувствительность с дополнительной неопределённостью USD/CNH |

Core: `open`, `close`, `high`, `low`, `volume`, `cbr_usdrub_effective`,
`swaprate_prev_session`, `key_rate_pct_effective`, `spread_close_minus_cbr`.
Дополнения: `ruonia_pct_last_reported`, `brent_eia_spot_usd_bbl_lag10d`,
`usdcnh_investing_price_prev_date` — только при явном выборе B1/B2.
Число признаков включает известный `close_t`, исключает ключ `date`.
Единицы и экономический смысл сохранены из [полного словаря B2](feature_data_dictionary.md).

`swaprate_current` исключён: общая информация о публикации funding не доказывает
доступность всех исторических значений к origin. `value` постоянно равно нулю,
его единица не установлена. Оба поля отсутствуют в store и metadata как значения.
`begin/end`, даты, страницы PDF и `swaprate_available_assumed` не предикторы.
Закрытый allowlist в коде и конфигурации отклоняет лишние названия, будущие
target-колонки, изменение default на чувствительный сценарий и общую нормализацию.

## Метаданные и связь по date

Оба CSV сохраняют исходный порядок и имеют уникальный строго возрастающий `date`.
`source_row_number` — номер строки данных v2 от 1, без строки заголовка.
Никакой переиндексации на календарные дни или сортировки с исправлением входа нет:
неверный порядок, дубликаты или расхождение с календарём вызывают ошибку.

Скопированы из v2 без изменения:

- `begin`, `end` — исходные метки свечи, не фактическое время получения;
- `cbr_effective_date`, `cbr_rate_age_calendar_days` — действие курса и возраст;
- `swaprate_source_trade_date`, `swaprate_available_assumed` — предыдущая торговая
  дата funding и признак его заполненности, не подтверждение PIT;
- `key_rate_reference_date`, `key_rate_pdf_source_page` — дата записи и provenance;
- `ruonia_reference_date`, `ruonia_source_dateupdate`, `ruonia_age_calendar_days`;
- `brent_eia_reference_date`, `brent_eia_age_calendar_days`;
- `usdcnh_investing_reference_date`, `usdcnh_investing_age_calendar_days`.

Добавлены только служебные поля: `source_dataset`, `source_sha256`,
`source_row_number`, `schema_version`, `lag_policy_version`, `pit_status`,
`price_type=CANDLE_CLOSE`, `missing_features` (имена пустых полей store через `|`).
Публикационные timestamps не выдумываются. Метаданные могут иметь пустые поля:
например, у первой строки нет предыдущей торговой даты.
Ссылки на источники каждого экономического ряда остаются в
[каталоге B2](feature_availability_catalog.csv), не дублируются как новые предикторы.

## Пропуски и preprocessing

В каждом сценарии 154 пропуска `swaprate_prev_session` — **13.275862%** строк.
Все остальные экономические значения заполнены. Все 1160 строк сохранены;
1006 полностью заполненных строк — описание покрытия, не автоматический фильтр.
Пропуск в CSV — пустая строка; существующий ноль — наблюдаемое значение.
Нет заполнения нулём/средним/медианой, bfill, интерполяции, scaler или PCA.

C выбирает модельную обработку пропусков внутри train/walk-forward совместно
с A. Scaler, PCA, импутация, feature selection должны обучаться только на
разрешённом обучающем окне каждого origin. Fit на всём периоде использовал бы
информацию о распределении будущих значений. Параметры не вычисляются в B3.
Новые изменения/доходности/funding deltas/индикаторы также не создаются.

## Пример загрузки без обучения

Из корня репозитория, с обычным Python:

```python
import math
from scripts.build_model_features import load_config, load_scenario

scenario = "core"  # B1/B2 включаются только явным другим ключом
config = load_config()
columns = config["scenarios"][scenario]["features"]
rows = load_scenario(scenario=scenario)  # date + только выбранные значения
dates = [row["date"] for row in rows]
X = [[float(row[name]) if row[name] != "" else math.nan
      for name in columns] for row in rows]
# NaN сохраняет пропуск; это не импутация. Fit/scoring здесь не выполняется.
```

`load_scenario()` возвращает строки, сохраняя точное исходное представление.
При join с metadata проверить `one-to-one` по date и совпадение порядка/числа строк.
Метаданные не добавлять в X автоматически. Ни config, ни загрузчик не выбирают
сценарий по метрике и не запускают бэктест.

## Использование CLOSE и оставшиеся ограничения

Для origin t известен `close_t` только после полной свечи. Это и уровень для
RW/регрессионных преобразований, и наблюдение ряда будущего target, но **не**
готовый supervised target. `close_{t+h}` формирует backtest pipeline по h-й
следующей торговой дате USDRUBF; строки без доступной метки остаются unscored.
Не использовать обычный календарный shift, future target как predictor или
пропускать торговые даты ради смены горизонта. Обучающая h-шаговая метка доступна
лишь после фактического появления соответствующего будущего CLOSE.

Остаются ограничения B1/B2: неизвестный historical available_at полной свечи,
предварительный календарь, историческая доступность предыдущего funding,
действующий курс/ставка вместо последнего объявленного будущего уровня.
Для B1 добавляются DateUpdate/календарь ЦБ и недоказанная достаточность Brent lag10;
для B2 — тип Price, timezone/cut-off, исправления и аномалии USD/CNH.
Проверки дат и согласование A не повышают assumed/unknown до verified.
Origin по часам, новая обработка пропусков, новые производные и преобразования
остаются решениями B/C. Список и сценарии повторного согласования не требуют,
пока сохраняется утверждённая в B3 схема.

Проверки без обучения:

```powershell
python -m unittest discover -s tests -p "test_model_features.py" -v
```

Исходные v2/raw, календарь, протокол, benchmark-прогнозы и закрытый final test
не изменены. Модели/PCA/Ridge/DFM не обучались; B4 не запускался.
