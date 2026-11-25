"""Figures for notebooks/model_comparison_2025.ipynb (validation 2025 targets only)."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL_TEST_START = pd.Timestamp('2026-01-01')
SOURCE_LABEL = {'model': 'модельная дисперсия', 'model_quantiles': 'модельные квантили',
                'conformal': 'конформный интервал'}


def model_colors(models):
    palette = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
    return {m: ('#7f7f7f' if m == 'RW' else palette[i % len(palette)]) for i, m in enumerate(models)}


def _guard(frame, column='target_date'):
    if (pd.to_datetime(frame[column]) >= FINAL_TEST_START).any():
        raise ValueError('Refusing to plot 2026 dates')
    return frame


def _format_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(alpha=0.3)


def panel_title(label, h, stats):
    text = f'{label}, h={h}: RMSE={stats["rmse"]:.4f}, RMSE/RW={stats["ratio"]:.4f}'
    if stats.get('coverage_90') is not None and np.isfinite(stats['coverage_90']):
        text += f'\nпокрытие 90%/95% ({stats["source"]}): {stats["coverage_90"]:.1%} / {stats["coverage_95"]:.1%}'
    return text


def draw_panel(ax, data, actual, color, label, h, stats, source=None):
    """Actual close (black), forecast at its target date, optional 90%/95% bands."""
    data = _guard(data.sort_values('target_date'))
    ax.plot(actual.index, actual.values, color='black', lw=1.3, label='Факт: CLOSE USDRUBF')
    if source is not None and data[['lower_95', 'upper_95']].notna().any().all():
        src = SOURCE_LABEL[source]
        ax.fill_between(data.target_date, data.lower_95, data.upper_95, color=color, alpha=0.13, lw=0,
                        label=f'95% — {src}')
        ax.fill_between(data.target_date, data.lower_90, data.upper_90, color=color, alpha=0.30, lw=0,
                        label=f'90% — {src}')
    ax.plot(data.target_date, data.prediction, color=color, lw=1.1, label=f'Прогноз: {label}')
    ax.set_title(panel_title(label, h, stats), fontsize=9)
    ax.set_ylabel('RUB за USD')
    _format_dates(ax)


def model_figure(data, actual, label, horizons, stats, color, source=None):
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.3 * len(horizons)), sharex=True, squeeze=False)
    for ax, h in zip(axes[:, 0], horizons):
        draw_panel(ax, data.loc[data.horizon.eq(h)], actual, color, label, h, stats[h], source)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('target_date, 2025')
    fig.tight_layout()
    return fig


def zoom_window(actual, sessions=63):
    """Window of ~3 months with the largest total absolute daily movement of the actual close."""
    movement = actual.diff().abs().rolling(sessions).sum()
    end = movement.idxmax()
    start = actual.index[actual.index.get_loc(end) - sessions + 1]
    return start, end


def all_models_figure(predictions, actual, h, models, labels, colors, zoom):
    data = _guard(predictions.loc[predictions.horizon.eq(h) & predictions.model_id.isin(models)])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw=dict(height_ratios=[1.3, 1]))
    for ax, (start, end) in ((top, (actual.index.min(), actual.index.max())), (bottom, zoom)):
        seg = actual.loc[start:end]
        ax.plot(seg.index, seg.values, color='black', lw=2.6, label='Факт: CLOSE USDRUBF', zorder=5)
        for i, model in enumerate(models):
            m = data.loc[data.model_id.eq(model) & data.target_date.between(start, end)].sort_values('target_date')
            ax.plot(m.target_date, m.prediction, color=colors[model], lw=0.9,
                    ls=('-' if i % 2 == 0 else '--'), label=labels[model])
        ax.set_ylabel('RUB за USD')
        _format_dates(ax)
    top.set_title(f'Все модели, h={h}: прогноз в точке target_date (validation 2025)')
    bottom.set_title(f'Увеличение: {zoom[0].date()} — {zoom[1].date()} (наибольшее суммарное движение курса)')
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    top.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1)
    fig.tight_layout()
    return fig


def rmse_ratio_bars(metrics, models, labels, cap=1.12):
    pivot = metrics.pivot(index='model_id', columns='horizon', values='rmse_ratio_vs_rw').loc[models]
    fig, ax = plt.subplots(figsize=(17, 6))
    width = 0.8 / pivot.shape[1]
    x = np.arange(len(models))
    cmap = plt.get_cmap('viridis')
    for j, h in enumerate(pivot.columns):
        values = pivot[h].to_numpy()
        bars = ax.bar(x + (j - 3) * width, np.minimum(values, cap), width, color=cmap(j / 6), label=f'h={h}')
        for bar, value in zip(bars, values):
            if value > cap:
                ax.text(bar.get_x() + bar.get_width() / 2, cap, f'{value:.2f}', rotation=90, fontsize=6,
                        ha='center', va='top', color='white')
    ax.axhline(1.0, color='black', lw=1.2)
    low = max(0.0, min(0.95, np.nanmin(pivot.to_numpy()) - 0.01))
    ax.set_ylim(low, cap)
    ax.set_xticks(x, [labels[m] for m in models], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('RMSE / RMSE(RW)')
    ax.set_title(f'RMSE/RW по моделям и горизонтам, validation 2025 (столбцы выше {cap} обрезаны, значение подписано)')
    ax.legend(ncol=7, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig
