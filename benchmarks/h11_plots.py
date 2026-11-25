"""Figures for H11-REL-01 (validation 2025 only)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'
COLORS = {'R0_MATCHED': '#257de0', 'R1': '#f47230', 'RW': '#9c9c9c', 'actual': INK}
LABELS = {'R0_MATCHED': 'R0: CatBoost Core v2 (10)', 'R1': 'R1: Core v2 + 6 отн. признаков',
          'RW': 'RW', 'actual': 'Факт (CLOSE USDRUBF)'}
DPI = 150
TEST_START = pd.Timestamp('2026-01-01')


def _style(ax):
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#bdbdbd')
    ax.tick_params(colors=INK2, labelsize=8)


def _guard(dates):
    if (pd.to_datetime(dates) >= TEST_START).any():
        raise ValueError('Final test 2026 must not be plotted')


def zoom_window(group, length=63):
    """Window of `length` targets with the largest total |daily change| of the actual."""
    actual = group.sort_values('target_date').actual.to_numpy()
    moves = np.abs(np.diff(actual, prepend=actual[0]))
    sums = np.convolve(moves, np.ones(length), 'valid')
    start = int(np.argmax(sums))
    dates = group.sort_values('target_date').target_date.to_numpy()
    return dates[start], dates[start+length-1]


def _lines(ax, group):
    group = group.sort_values('target_date')
    ax.plot(group.target_date, group.actual, color=COLORS['actual'], lw=1.6, label=LABELS['actual'], zorder=4)
    ax.plot(group.target_date, group.RW, color=COLORS['RW'], lw=1.2, ls='--', label=LABELS['RW'], zorder=2)
    for arm in ('R0_MATCHED', 'R1'):
        ax.plot(group.target_date, group[arm], color=COLORS[arm], lw=1.4, label=LABELS[arm], zorder=3)
    ax.set_ylabel('руб. за 1 USD (цена USDRUBF)', color=INK2, fontsize=8)
    ax.set_xlim(group.target_date.min(), group.target_date.max())  # no axis tick past 2025-12-30
    _style(ax)


def forecast_figure(scored, h, metrics, path):
    group = scored[scored.horizon == h]
    _guard(group.target_date)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(height_ratios=[1.3, 1]))
    _lines(top, group)
    m = metrics[metrics.horizon == h].set_index('model_id').rmse_ratio_vs_rw
    top.set_title(f'h = {h}: прогноз CLOSE[t+h] по дате цели, validation 2025   '
                  f'RMSE/RW: R0 {m["R0_MATCHED"]:.4f}, R1 {m["R1"]:.4f}', loc='left', fontsize=10, color=INK)
    start, end = zoom_window(group)
    top.axvspan(start, end, color='#f2f2f2', zorder=0)
    top.legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    _lines(bottom, group[(group.target_date >= start) & (group.target_date <= end)])
    bottom.set_title(f'Увеличение: {pd.Timestamp(start).date()} – {pd.Timestamp(end).date()} '
                     '(63 сессии с наибольшим суммарным |Δ|)', loc='left', fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def forecast_panels(scored, horizons, metrics, path):
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3.2*len(horizons)), sharex=True)
    for ax, h in zip(axes, horizons):
        group = scored[scored.horizon == h]
        _guard(group.target_date)
        _lines(ax, group)
        m = metrics[metrics.horizon == h].set_index('model_id').rmse_ratio_vs_rw
        ax.set_title(f'h = {h}   RMSE/RW: R0 {m["R0_MATCHED"]:.4f}, R1 {m["R1"]:.4f}', loc='left', fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def rmse_ratio_bars(metrics, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    horizons = sorted(metrics.horizon.unique())
    x = np.arange(len(horizons))
    width = .36
    for k, arm in enumerate(('R0_MATCHED', 'R1')):
        values = metrics[metrics.model_id == arm].set_index('horizon').loc[horizons].rmse_ratio_vs_rw.to_numpy()
        bars = ax.bar(x+(k-.5)*(width+.02), values-1, width, bottom=1, color=COLORS[arm], label=LABELS[arm])
        for bar, value in zip(bars, values):
            y = value+(.0004 if value >= 1 else -.0004)
            ax.text(bar.get_x()+bar.get_width()/2, y, f'{value:.4f}', ha='center',
                    va='bottom' if value >= 1 else 'top', fontsize=7, color=INK2)
    ax.axhline(1, color=INK, lw=1)
    ax.set_xticks(x, [f'h={h}' for h in horizons])
    ax.set_ylabel('RMSE / RMSE(RW)', color=INK2, fontsize=8)
    low = min(.99, metrics[metrics.model_id.isin(['R0_MATCHED', 'R1'])].rmse_ratio_vs_rw.min()-.003)
    high = max(1.01, metrics[metrics.model_id.isin(['R0_MATCHED', 'R1'])].rmse_ratio_vs_rw.max()+.003)
    ax.set_ylim(low, high)
    ax.set_title('RMSE относительно RW, validation 2025 (ниже 1 — лучше RW)', loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def hac_difference(hac, path):
    primary = hac[hac.family == 'primary'].sort_values('horizon')
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = primary.horizon.to_numpy()
    ax.errorbar(x, primary.mean_squared_loss_difference,
                yerr=[primary.mean_squared_loss_difference-primary.ci_low,
                      primary.ci_high-primary.mean_squared_loss_difference],
                fmt='o', color=COLORS['R1'], ecolor=COLORS['R1'], ms=8, capsize=4, lw=2,
                markeredgecolor='white', markeredgewidth=2)
    ax.axhline(0, color=INK, lw=1)
    top = primary.ci_high.max()
    for _, row in primary.iterrows():
        ax.text(row.horizon, top+(top-primary.ci_low.min())*.06, f'p={row.p_two_sided:.2f}\nHolm {row.holm_p:.2f}',
                ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(x, [f'h={h}' for h in x])
    ax.set_ylabel('средняя d = e²(R1) − e²(R0), руб.²', color=INK2, fontsize=8)
    ax.set_title('R1 − R0: разность квадратов ошибок и 95% HAC-интервал (ниже 0 — R1 лучше)',
                 loc='left', fontsize=10, color=INK)
    ax.margins(y=.25)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def importance_bars(importance, added, path):
    r1 = importance[importance.model_id == 'R1'].groupby('feature')['mean'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [COLORS['R1'] if f in added else COLORS['R0_MATCHED'] for f in r1.index]
    ax.barh(r1.index, r1.values, color=colors, height=.6)
    for y, value in enumerate(r1.values):
        ax.text(value+r1.max()*.01, y, f'{value:.1f}', va='center', fontsize=7, color=INK2)
    ax.set_xlabel('PredictionValuesChange, среднее по origin и h', color=INK2, fontsize=8)
    ax.set_title('R1: важность признаков (оранжевые — 6 новых, синие — Core v2)', loc='left', fontsize=10, color=INK)
    _style(ax)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
