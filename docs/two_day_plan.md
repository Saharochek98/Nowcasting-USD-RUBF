# План команды — следующий рабочий день (24.11.2026)

**Актуализирован:** 23.11.2026 · **USDRUBF**, daily CANDLE_CLOSE, `h=1..7` торговых сессий.  
**Факт:** B1–B5 подготовлены и объединены в `main`; [передача данных A/C/D](data_handoff_to_modeling.md) завершена. Календарь `PROVISIONAL`, режим `conditional EOD / pseudo-PIT`, финальный test 2026 закрыт для model selection.

| Владелец | Приоритет | Действие 24.11 | DoD / что принести на review |
|---|---|---|---|
| **A — research lead** | P0 | Контрольный clean rerun RW/AR; зафиксировать Git SHA, протокол созревания h-step target, общие validation keys и доступные feature-сценарии | Новый manifest и validation comparison с числом общих пар; отсутствие доступа к final test |
| **C — modeling** | P0 | `core` из `load_scenario()`: price-only и Ridge; train-only imputation/scaling, параметры только по прошлым созревшим меткам | OOS validation-прогнозы h=1…7, RW comparison, reproducible конфиг и leakage tests |
| **D — uncertainty** | P0 | Интервальный benchmark RW 90%/95% с калибровкой только по исторически созревшим ошибкам | coverage/width/interval score на validation по h и n |
| **B — data** | При необходимости | Проверить, что A/C/D получили [инструкцию](data_handoff_to_modeling.md); помочь с 154 initial funding missing, 198 observed zero, календарём/метаданными и PIT-допущениями | Короткое подтверждение принятого интерфейса; новые данные/аудиты **не требуются** |
| **A/C (при готовности)** | P1 | Решить, нужен ли PCA+Ridge и минимальный DFM/Kalman после оценки Ridge; отдельно preregister чувствительность S0/S1/S2 | Явное go/no-go и список проверок без test |

## На review проверить

- Сравнения точечных моделей ведутся по одним `(origin,target,h)`; будущие метки не участвуют в fit.
- `core` = 9 признаков. RUONIA/Brent (11) и USD/CNH (12) — только отдельно согласованные sensitivity.
- 154 пропуска предыдущего funding не заменяются нулями; 198 наблюдаемых нулей не превращаются в missing. S2 с декабря 2022 — не замена основного эксперимента, training samples могут различаться.
- Предварительный календарь и исторический `available_at` **не** сертифицированы, несмотря на завершение аудитов. Финальный test 2026 не используется в настройке.
- Отделять `IMPLEMENTED` от `VALIDATION_EVALUATED` и `TEST_CONFIRMED`. Если DFM не даёт измеримого выигрыша на validation, допустимо оставить более простой Ridge/RW.

**История B и детали:** [Project Status](project_status.md), [B1/B2/B3/B4 материалы](data_handoff_to_modeling.md). Ранее выполненные задачи не переписаны и не дублируются.
