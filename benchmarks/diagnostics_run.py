"""Run saved RW/AR diagnostics with python -m benchmarks.diagnostics_run."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .core import load_calendar, load_config, load_data
from .diagnostics import (align_losses, drift_decomposition, historical_series,
                          rolling_volatility, saved_parameters, serial_diagnostics,
                          summarize_losses, train_volatility_thresholds, volatility_diagnostics)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Origin (session date)")


def make_figures(output, config, params, correlations, losses, loss_summary, regimes, drift, drift_summary):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.constrained_layout.use": True})
    names = []

    def save(fig, name):
        fig.savefig(output / name, dpi=config["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for ax, coefficient in zip(axes[:2], ("phi", "c")):
        ax.plot(params.origin_date, params[coefficient], lw=1.1)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(pd.Timestamp("2025-01-01"), color="tab:orange", ls="--", label="Validation starts")
        ax.set_ylabel(coefficient)
        ax.set_title(f"Saved expanding-window {coefficient}; one observation per origin")
        _date_axis(ax)
        ax.legend()
    parameter_bins = np.histogram_bin_edges(params.phi, bins=config["histogram_bins"])
    axes[2].hist(params.phi, bins=parameter_bins, alpha=0.55, density=True, label="All available train/validation origins")
    axes[2].hist(params.loc[params.used_for_validation, "phi"], bins=parameter_bins, alpha=0.55,
                 density=True, label="Origins used for validation forecasts")
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set(xlabel="phi", ylabel="Density", title="Saved phi distribution (not significance tests)")
    axes[2].legend()
    save(fig, "ar1_coefficients.png")

    for split in ("train", "validation"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [("change", "acf", "Price differences: ACF"), ("change", "pacf", "Price differences: PACF (Yule-Walker MLE)"),
                  ("log_return", "acf", "Log returns: ACF"), ("abs_log_return", "acf", "Absolute log returns: ACF")]
        for ax, (series, statistic, title) in zip(axes.flat, panels):
            frame = correlations[correlations.split.eq(split) & correlations.series.eq(series)]
            for segment, values in frame.groupby("segment"):
                values = values[values.lag.gt(0)]
                marker, stems, _ = ax.stem(values.lag, values[statistic], basefmt=" ", label=f"Segment {segment}, n={values.n.iloc[0]}")
                marker.set_markersize(3)
                reference = 1.96 / np.sqrt(values.n.iloc[0])
                ax.axhline(reference, color="gray", ls=":", lw=0.7)
                ax.axhline(-reference, color="gray", ls=":", lw=0.7)
            ax.axhline(0, color="black", lw=0.7)
            ax.set(title=title, xlabel="Session lag", ylabel=statistic.upper())
            ax.legend(fontsize=8)
        fig.suptitle(f"{split}: contiguous segments only; dotted lines are pointwise white-noise references")
        save(fig, f"autocorrelation_{split}.png")

    for h in config["plot_horizons"]:
        frame = losses[losses.horizon.eq(h)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        error_bins = np.histogram_bin_edges(np.r_[frame.error_rw, frame.error_ar], bins=config["histogram_bins"])
        for model, label in (("rw", "RW"), ("ar", "AR(1)")):
            axes[0, 0].plot(frame.origin_date, frame[f"error_{model}"], label=label, lw=0.9, alpha=0.8)
            axes[0, 1].hist(frame[f"error_{model}"], bins=error_bins, alpha=0.5, label=label)
        axes[0, 0].axhline(0, color="black", lw=0.7)
        axes[0, 0].set(ylabel="Prediction - actual", title="Walk-forward errors")
        _date_axis(axes[0, 0])
        axes[0, 0].legend()
        axes[0, 1].set(xlabel="Prediction - actual", ylabel="Count", title="Error distributions")
        axes[0, 1].legend()
        axes[1, 0].plot(frame.origin_date, frame.cumulative_d, color="tab:purple")
        axes[1, 0].axhline(0, color="black", lw=0.7)
        axes[1, 0].set(ylabel="Cumulative (AR squared error - RW squared error)", title="Positive cumulative loss favors RW")
        _date_axis(axes[1, 0])
        axes[1, 1].scatter(frame.error_rw, frame.error_ar, s=12, alpha=0.6)
        limits = [min(frame.error_rw.min(), frame.error_ar.min()), max(frame.error_rw.max(), frame.error_ar.max())]
        axes[1, 1].plot(limits, limits, color="black", ls="--", lw=0.8)
        axes[1, 1].set(xlabel="RW error", ylabel="AR(1) error", title="Identical origin/target pairs")
        fig.suptitle(f"Validation errors, h={h}, n={len(frame)}")
        save(fig, f"errors_h{h}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(loss_summary.horizon, loss_summary.mean_d,
                yerr=np.vstack([loss_summary.mean_d - loss_summary.ci_lower, loss_summary.ci_upper - loss_summary.mean_d]),
                fmt="o", capsize=5, label="Mean loss difference, pointwise 95% HAC CI")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Horizon (sessions)", ylabel="Mean AR squared error - RW squared error",
           title="Fixed Bartlett/Newey-West lag max(20, h-1); no automatic bandwidth search")
    ax.legend()
    save(fig, "hac_loss_intervals.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["low", "medium", "high"]
    for i, h in enumerate(config["plot_horizons"]):
        frame = regimes[regimes.horizon.eq(h)].set_index("regime").reindex(labels)
        axes[0].plot(labels, frame.rmse_ratio, marker="o", label=f"h={h}")
        axes[1].bar(np.arange(3) + (i - 1.5) * 0.18, frame.n, width=0.18, label=f"h={h}")
    axes[0].axhline(1, color="black", ls="--", lw=0.8)
    axes[0].set(ylabel="RMSE AR / RW", title="Train-defined 20-session volatility groups")
    axes[1].set(xticks=np.arange(3), xticklabels=labels, ylabel="Validation pairs", title="Sample sizes; groups are not randomized")
    for ax in axes:
        ax.legend()
    save(fig, "volatility_regimes.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, h in zip(axes.flat, config["plot_horizons"]):
        frame = drift[drift.horizon.eq(h)]
        ax.plot(frame.origin_date, frame.constant_component, label="Constant component", lw=1)
        ax.plot(frame.origin_date, frame.autoregressive_component, label="Lagged-change component", lw=1)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(title=f"h={h}: contributions to predicted level change", ylabel="Price change")
        _date_axis(ax)
        ax.legend()
    save(fig, "drift_decomposition.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(drift_summary.horizon, drift_summary.mean_abs_constant, marker="o", label="Constant")
    axes[0].plot(drift_summary.horizon, drift_summary.mean_abs_autoregressive, marker="o", label="Lagged change")
    axes[0].set(xlabel="Horizon", ylabel="Mean absolute level contribution", title="Component magnitudes")
    axes[1].bar(drift_summary.horizon - 0.18, drift_summary.mean_constant_loss_allocation, width=0.36, label="Constant")
    axes[1].bar(drift_summary.horizon + 0.18, drift_summary.mean_autoregressive_loss_allocation, width=0.36, label="Lagged change")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(xlabel="Horizon", ylabel="Mean loss allocation", title="Algebraic allocation, shared cross term; not causal")
    for ax in axes:
        ax.legend()
    save(fig, "drift_summary.png")
    return names


def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(root, output, config, params, stats, correlations, lb, segments, loss_summary, regimes, drift_summary, thresholds, quarters, influential, calibration_end):
    validation_phi = stats[(stats.scope == "validation_forecast_origins") & (stats.coefficient == "phi")].iloc[0]
    validation_c = stats[(stats.scope == "validation_forecast_origins") & (stats.coefficient == "c")].iloc[0]
    h1, h7 = loss_summary.set_index("horizon").loc[1], loss_summary.set_index("horizon").loc[7]
    drift7 = drift_summary.set_index("horizon").loc[7]
    quarter7 = quarters[quarters.horizon.eq(7)]
    first_quarter = quarter7.loc[quarter7.target_quarter.eq("2025Q1"), "sum_d"].iloc[0]
    signed_p = lb[lb.series.isin(["change", "log_return"])].groupby("split").p_value.min()
    absolute_p = lb[lb.series.eq("abs_log_return")].groupby("split").p_holm.min()
    plots = output.relative_to(root).as_posix()
    lines = ["# Диагностика Random Walk и AR(1): train и validation", "",
             "## Цель, данные и ограничения", "",
             "Объяснить отсутствие выигрыша AR(1) на разностях CLOSE относительно RW в validation за 2025 год.",
             "Это исследование сохранённых прогнозов, а не настройка моделей. Новые модели не обучались.",
             "Использованы исходный MOEX ISS CANDLE_CLOSE, календарь reconstructed_moex_calendar_provisional",
             "и артефакты results/benchmark_calendar_v2. Календарная интеграция A2 завершена: 1 668 календарных",
             "дат, 1 160 ожидаемых и наблюдаемых торговых дат, 0 пропусков/неожиданных свечей; рабочие субботы сохранены.",
             "Все 254 validation-пары каждого горизонта и метрики совпали с A1; старые файлы не изменены.",
             "Train: 683 цены (2022-04-26–2024-12-30), validation: 254 цены (2025-01-03–2025-12-30).",
             "Цены final test отрезаны до любых диагностических преобразований; final test не оценивается.",
             "Сохраняются conditional EOD / pseudo-PIT, предварительность календаря и неопределённость исторического",
             "времени публикации свечей. Наблюдаемая candle date не доказывает доступность в конкретную минуту.", "",
             "## Методика, зафиксированная до расчёта", "",
             "- ACF/PACF: лаги 1–20; ACF с общим знаменателем n, PACF Yule–Walker MLE (`ywm`).",
             "- Ljung–Box: лаги 5/10/20, model_df=0 для исходных рядов, без автоматического выбора лагов.",
             "- Train и validation анализируются отдельно; разность/доходность первой validation-сессии может",
             "  использовать последнюю train-цену. Ряды разделяются на непрерывные сегменты при пропусках;",
             "  сегменты короче 42 наблюдений не тестируются и не склеиваются.",
             "- HAC для среднего d: Bartlett/Newey–West, L=max(20,h−1), поправка n/(n−1), нормальный 95% интервал.",
             "  Лаг покрывает перекрытие горизонтов и фиксирован заранее. Это асимптотические точечные интервалы,",
             "  не одновременные; ограниченная мощность и оценивание коэффициентов моделей сохраняются как ограничения.",
             "  При пропусках origin pooled HAC не рассчитывается. Holm p-values дополнительно показаны",
             "  для семейства всех Ljung–Box тестов и отдельно для семи HAC-сравнений; результаты исследовательские.",
             "- Волатильность: std последних 20 log-доходностей, ddof=1, включая только данные по origin.",
             "  Границы — линейные train-терцили, затем неизменны; группы n<30 явно отмечены.",
             "- Ошибки prediction−actual и d=e_AR²−e_RW² сравниваются на одинаковых origin/target/horizon.",
             "  Все семь горизонтов участвуют в HAC; детальные графики — h=1,3,5,7. Изменений benchmark-правил нет.", "",
             "Реализации и определения: [ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html),",
             "[PACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html),",
             "[Ljung–Box](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html),",
             "[HAC](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).", "",
             "## Сохранённые коэффициенты", "",
             "Каждый origin учитывается один раз, без семикратного дублирования по h. Отдельно показаны все доступные",
             "train/validation origins и origins, использованные хотя бы для одного validation-target (включая конец train).", "",
             markdown_table(stats), "",
             f"На validation-forecast origins phi: среднее {validation_phi['mean']:.6g}, диапазон "
             f"[{validation_phi['min']:.6g}, {validation_phi['max']:.6g}], смен ненулевого знака {validation_phi.nonzero_sign_changes:.0f}.",
             f"Константа c: среднее {validation_c['mean']:.6g}, диапазон [{validation_c['min']:.6g}, {validation_c['max']:.6g}].",
             "Смена знака не доказывает смену рыночного режима; значимость отдельных OLS-коэффициентов не проверялась.", "",
             f"![Коэффициенты и гистограмма phi](../{plots}/ar1_coefficients.png)", "",
             "## Серийная зависимость и волатильность", "", markdown_table(lb), "",
             "Ниже описаны все фактически рассчитанные тесты. Малые p-values трактуются как свидетельство против",
             "отсутствия серийной корреляции в исследуемом сегменте, а не как доказанный OOS-выигрыш данной AR(1).",
             "Автокорреляция signed returns относится к линейной зависимости изменений; автокорреляция абсолютных",
             "доходностей относится к зависимости масштаба движений. Кластеризация волатильности сама по себе не",
             "означает предсказуемость направления и улучшение условного среднего цены.", ""]
    lines += [f"Минимальные необработанные p-values для changes/log returns: train={signed_p['train']:.4g}, "
              f"validation={signed_p['validation']:.4g}. На заданных лагах нулевая гипотеза отсутствия корреляции",
              "не отвергается при 5%; это не доказательство полного отсутствия предсказуемости.",
              f"Для абсолютных log returns минимальные Holm-p: train={absolute_p['train']:.4g}, "
              f"validation={absolute_p['validation']:.4g}. Это свидетельство кластеризации масштаба движений",
              "в этих выборках, а не доказательство полезности данной AR для направления цены. Обычный Ljung–Box",
              "не является универсальной проверкой условного среднего при произвольной гетероскедастичности.", ""]
    for split in ("train", "validation"):
        subset = correlations[(correlations.split == split) & (correlations.lag == 1)]
        lines += [f"### {split}", "", markdown_table(subset[["series", "n", "acf", "pacf"]]), "",
                  f"![ACF/PACF {split}](../{plots}/autocorrelation_{split}.png)", ""]
    lines += ["Обработка календаря/сегментов:", "", markdown_table(segments), "",
              "## Ошибки, накопленные потери и устойчивость", "",
              "Положительное d означает меньшую квадратичную ошибку RW. Положительная конечная сумма D не означает,",
              "что AR хуже на каждой дате; доля d>0 и квартальные суммы показывают эпизодичность различий.", "",
              markdown_table(loss_summary[["horizon", "n", "mean_d", "ci_lower", "ci_upper", "p_normal", "p_holm",
                                           "rw_better_share", "bias_rw", "bias_ar", "error_correlation", "top5_absolute_d_share"]]), "",
              f"![HAC-интервалы средней разницы потерь](../{plots}/hac_loss_intervals.png)", "",
              "Доля top5 относится к сумме |d| пяти наиболее крупных по |d| пар относительно всех |d|, а не к доле RMSE.",
              "Пары не удаляются. Конкретные даты, ошибки и вклады сохранены в influential_errors.csv.",
              "Квартальные суммы (по target_date) позволяют увидеть, равномерно ли накапливалась разница:", "",
              markdown_table(quarters), ""]
    lines += [f"Для h=7 суммарное d={h7.cumulative_d:.6g}, из него I квартал даёт {first_quarter:.6g} "
              f"({first_quarter / h7.cumulative_d:.1%} от итоговой чистой разницы). В III квартале сумма d отрицательна",
              "и AR оказывается точнее по суммарной квадратичной потере, поэтому ухудшение не равномерно во времени.",
              f"Пять крупнейших |d| при h=7 дают {h7.top5_absolute_d_share:.1%} всей суммы |d| и "
              f"{h7.top5_absolute_d_net:.6g} чистой разницы: крупные ошибки влияют, но не объясняют её полностью.",
              f"Доля пар, где RW точнее: h=1 {h1.rw_better_share:.1%}, h=7 {h7.rw_better_share:.1%}.",
              "Пять крупнейших по |d| пар для h=7:", "",
              markdown_table(influential[influential.horizon.eq(7)].head(5)[["origin_date", "target_date", "error_rw", "error_ar", "d"]]), ""]
    for h in config["plot_horizons"]:
        lines += [f"![Ошибки и накопленная разница потерь h={h}](../{plots}/errors_h{h}.png)", ""]
    excluded_zero = loss_summary[(loss_summary.ci_lower > 0) | (loss_summary.ci_upper < 0)].horizon.tolist()
    holm = loss_summary[loss_summary.p_holm.lt(0.05)].horizon.tolist()
    lines += [f"95% точечные HAC-интервалы исключают ноль на горизонтах: {excluded_zero}.",
              f"После Holm-коррекции p<0.05 для семи горизонтов: {holm}.",
              "Вывод ограничен 254 перекрывающимися парами на h, одним validation-годом и выбранным заранее HAC-лагом.",
              "Невыявление различия не доказывает эквивалентность моделей; интервал отражает неопределённость среднего d.", "",
              "## Разбиение по исторической волатильности", "",
              f"Train-границы sigma20: {thresholds[0]:.8f} и {thresholds[1]:.8f} (доли, не проценты).",
              "Эти границы не оценивались на validation и не используются для переключения моделей.", "",
              f"Пороги считаются доступными с {pd.Timestamp(calibration_end).date()}, последней использованной train-свечи.",
              "Origins до этой даты помечены calibration_unavailable: полные train-терцили тогда ещё недоступны.",
              "Такие строки не участвуют в сравнении трёх групп, но сохраняются в CSV и общих benchmark-метриках.",
              "Для h=1/3/5/7 исключено по этой причине соответственно 0/2/4/6 пар; это граница доступности,",
              "а не отбор по качеству. sigma20 на каждом origin в любом случае вычислена только по прошлым данным.", "",
              markdown_table(regimes), "", f"![Волатильностные группы](../{plots}/volatility_regimes.png)", "",
              "Различия ratio между группами — описательная проверка гипотезы зависимости относительного качества от",
              "исторической волатильности. Группы не рандомизированы, наблюдения зависимы и могут совпадать с отдельными",
              "периодами года; малые n ограничивают интерпретацию. Формальный тест взаимодействия не проводился.", "",
              "На h=1 ratio ниже единицы в низкой/средней группе и выше единицы в высокой. На h=3,5,7 ratio выше",
              "единицы во всех трёх группах. Строго монотонного ухудшения с ростом волатильности нет. Это даёт",
              "ограниченную описательную поддержку гипотезе неоднородности h=1, но не подтверждает устойчивое",
              "преимущество AR в отдельных режимах. В основных low/medium/high группах n>=30; строки",
              "unavailable отражают исключённые пары, а не дополнительные режимы. Наблюдения зависимы.", "",
              "## Рекурсивное разложение прогноза", "",
              "На каждом сохранённом origin: a0=DeltaC_t, b0=0; ak=phi*ak−1, bk=c+phi*bk−1.",
              "Суммы ak и bk дают соответственно вклад последней разности и константы в прогноз уровня.",
              "C_t + sum(ak+bk) проверено против сохранённого AR-прогноза; новая AR-модель не оценивалась.", "",
              markdown_table(drift_summary), "", f"![Компоненты прогноза](../{plots}/drift_decomposition.png)", "",
              f"![Масштабы и алгебраическое распределение потерь](../{plots}/drift_summary.png)", "",
              "Для константы A и AR-вклада B разница потерь раскладывается на",
              "(2*e_RW*A+A²+A*B) + (2*e_RW*B+B²+A*B). Это симметричное распределение перекрёстного члена,",
              "не причинная оценка и не backtest вариантов без константы. Оно показывает механизм текущего прогноза.", "",
              f"На h=7 средний вклад константы в уровень равен {drift7.mean_constant:.6g}, AR-части — "
              f"{drift7.mean_autoregressive:.6g}; константа больше по модулю на {drift7.constant_dominant_share:.1%} пар.",
              f"Bias RW={h7.bias_rw:.6g}, AR={h7.bias_ar:.6g}: дополнительный положительный сдвиг равен "
              f"{h7.bias_ar - h7.bias_rw:.6g}, что совпадает с суммой средних компонентов.",
              f"Алгебраическое распределение среднего d: константа {drift7.mean_constant_loss_allocation:.6g}, "
              f"AR-часть {drift7.mean_autoregressive_loss_allocation:.6g}; сумма {h7.mean_d:.6g}.",
              "Таким образом, гипотеза о вкладе положительного дрейфа в ухудшение поддерживается разложением именно",
              "сохранённого прогноза. Это не доказывает, что удаление константы улучшит OOS-качество: такой модельный",
              "эксперимент здесь не проводился. При малом phi влияние последней разности быстро затухает,",
              "а вклад константы накапливается по h, что согласуется с ростом положительного смещения AR.", "",
              "## Выводы и следующие шаги", "",
              "Подтверждено: календарная интеграция воспроизводит benchmarks; AR имеет больший validation RMSE на всех h.",
              "Поддержаны в рамках диагностики: малые выборочные lag-1 корреляции изменений, кластеризация",
              "абсолютных доходностей, неравномерное во времени ухудшение и положительный вклад константы в разницу потерь.",
              "Не подтверждены: гипотеза о смене знака phi (он положителен на всех сохранённых origins),",
              "устойчивое превосходство текущей AR над RW, монотонная связь её относительного качества с волатильностью",
              "и семейно-значимое различие потерь при 5% после Holm-коррекции семи горизонтов.",
              "Коэффициенты, серийная зависимость, эпизодичность ошибок, волатильностные группы и вклады константы",
              "количественно описаны в таблицах выше. Численные изменения знака и групповое преимущество сами по себе",
              "не устанавливают рыночные режимы, причинность или устойчивое прогнозное преимущество.",
              "Проверка наличия полезной для условного среднего информации требует отдельного заранее описанного",
              "исследования и честной вневыборочной проверки. Текущие модели не изменены по результатам диагностики.",
              "Следующие шаги: завершить независимый аудит календаря/времени доступности; сформулировать до нового",
              "эксперимента гипотезы о стабильности среднего и дисперсии; при необходимости проверять модели дисперсии",
              "и взаимодействие с волатильностью отдельным протоколом. Сохранить final test закрытым до фиксации решений.", "",
              "## Воспроизводимость", "",
              "Запуск из корня: `python -m benchmarks.diagnostics_run` после установки requirements-diagnostics.txt.",
              "Настройки зафиксированы в configs/ar1_diagnostics.yaml; копия настроек, train-пороги, версии Python/библиотек,",
              "Git SHA, состояние рабочей папки и SHA256 входов/кода записаны в diagnostics_manifest.json.",
              "Исходные benchmark-артефакты проверяются по хешам до/после; actuals читаются только для validation.",
              "При запуске до коммита SHA указывает базовый коммит, а хеши — использованный незакоммиченный код.",
              "После разрешённого коммита можно повторить диагностику для записи нового SHA. Commit/push/merge не выполнялись."]
    (root / "reports/ar1_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ar1_diagnostics.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings_path = root / args.config
    with settings_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    benchmark = load_config(root / config["benchmark_config"])
    inputs, output = root / config["benchmark_run"], root / config["output"]
    if output.resolve() in (inputs.resolve(), (root / "results/benchmark_run").resolve()):
        raise ValueError("Diagnostic output must be separate from benchmark artifacts")
    if config["allowed_splits"] != ["train", "validation"]:
        raise ValueError("This diagnostic never accepts final_test")
    if (config["hac_kernel"] != "Bartlett" or not config["hac_small_sample_correction"]
            or config["hac_lag_rule"] != "max(base_lag, horizon - 1)"):
        raise ValueError("Expected the predeclared Bartlett HAC rule with small-sample correction")
    git_state = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    preserved = {p: _hash(p) for folder in (inputs, root / "results/benchmark_run") for p in folder.iterdir() if p.is_file()}
    manifest = json.loads((inputs / "run_manifest.json").read_text())
    if manifest["scored_splits"] != ["validation"] or manifest.get("final_test_used_for_fitting_or_scoring", True):
        raise ValueError("Expected validation-only A2 source run")
    data = load_data(root / benchmark["data_path"])
    calendar = load_calendar(data, benchmark, root)
    end = benchmark["splits"]["validation"][1]
    history = historical_series(data.loc[:end, "price"], calendar, end)
    del data  # final-test prices are not retained for any diagnostic operation
    forecasts = pd.read_parquet(inputs / "predictions_ar1.parquet")
    if (forecasts.origin_date > pd.Timestamp(end)).any():
        raise ValueError("Diagnostic parameter input includes final-test origins")
    scored = pd.read_parquet(inputs / "scored_predictions.parquet", filters=[("split", "==", "validation")])
    losses = align_losses(scored)
    params, statistics = saved_parameters(forecasts, losses.origin_date.unique(), benchmark)
    correlations, lb, segments = serial_diagnostics(history, benchmark["splits"], config)
    loss_summary = summarize_losses(losses, calendar, config)
    volatility = rolling_volatility(history.log_return, config["volatility_window"], config["volatility_ddof"])
    thresholds, threshold_n = train_volatility_thresholds(volatility, *benchmark["splits"]["train"], config["volatility_quantiles"])
    calibration_end = volatility.loc[pd.Timestamp(benchmark["splits"]["train"][0]):pd.Timestamp(benchmark["splits"]["train"][1])].last_valid_index()
    classified, regimes = volatility_diagnostics(losses, volatility, thresholds, config["plot_horizons"],
                                                config["minimum_regime_count"], calibration_end)
    drift, drift_summary = drift_decomposition(losses, params, history)
    quarterly = losses.assign(target_quarter=losses.target_date.dt.to_period("Q").astype(str)).groupby(
        ["horizon", "target_quarter"]).agg(n=("d", "size"), sum_d=("d", "sum"), mean_d=("d", "mean")).reset_index()
    influential = pd.concat([frame.assign(abs_d=frame.d.abs()).nlargest(10, "abs_d")
                            for _, frame in losses.groupby("horizon")], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"ar1_parameters": params, "parameter_statistics": statistics,
              "autocorrelations": correlations, "autocorrelation_tests": lb, "contiguous_segments": segments,
              "loss_differentials": losses, "loss_summary_hac": loss_summary, "loss_by_quarter": quarterly,
              "influential_errors": influential, "volatility_regimes": classified, "volatility_regime_metrics": regimes,
              "drift_decomposition": drift, "drift_summary": drift_summary}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "diagnostics_config.yaml").write_bytes(settings_path.read_bytes())
    figures = make_figures(output, config, params, correlations, losses, loss_summary, regimes, drift, drift_summary)
    write_report(root, output, config, params, statistics, correlations, lb, segments, loss_summary, regimes,
                 drift_summary, thresholds, quarterly, influential, calibration_end)
    if any(_hash(path) != original for path, original in preserved.items()):
        raise RuntimeError("Benchmark artifacts changed during diagnostics")
    sources = [settings_path, root / config["benchmark_config"], root / benchmark["data_path"],
               root / benchmark["calendar_path"], root / benchmark["full_calendar_path"],
               root / "requirements-diagnostics.txt", *sorted((root / "benchmarks").glob("*.py")),
               *sorted((root / "tests").glob("*.py"))]
    metadata = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    git_status_at_start=git_state, implementation_state="uncommitted_changes" if git_state else "committed",
                    python=platform.python_version(), packages={name: version(name) for name in
                    ("numpy", "pandas", "PyYAML", "pyarrow", "statsmodels", "matplotlib", "scipy", "pytest")},
                    sha256={p.relative_to(root).as_posix(): _hash(p) for p in sources},
                    preserved_benchmark_sha256={p.relative_to(root).as_posix(): value for p, value in preserved.items()},
                    diagnostic_splits=["train", "validation"], final_test_used=False,
                    history_end=str(history.index.max().date()), figures=figures,
                    volatility_thresholds=dict(low=float(thresholds[0]), high=float(thresholds[1]), train_n=threshold_n,
                                               train_end=benchmark["splits"]["train"][1], available_from=str(calibration_end.date()),
                                               excluded_precalibration_pairs=int(classified.regime.eq("calibration_unavailable").sum()),
                                               quantiles=config["volatility_quantiles"]),
                    coefficient_source="saved_origin_specific_walk_forward_estimates_no_refitting",
                    result_status="exploratory_conditional_EOD_pseudo_PIT_provisional_calendar")
    (output / "diagnostics_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Diagnostics completed; benchmarks preserved; final test unused.")
    print(statistics.to_string(index=False))
    print(lb.to_string(index=False))
    print(loss_summary.to_string(index=False))
    print(regimes.to_string(index=False))
    print(drift_summary.to_string(index=False))


if __name__ == "__main__":
    main()
