from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_CSV = REPO_ROOT / 'results/tables/knot_cross_domain_temporal_profiles_pilot64.csv'
OUT_DIR = REPO_ROOT / 'workshop'
OUT_PNG = OUT_DIR / 'fig_knot_domain_profiles.png'
OUT_PDF = OUT_DIR / 'fig_knot_domain_profiles.pdf'

ANCHORS = [10, 40, 70, 100]
COLORS = {'math': '#1f77b4', 'coding': '#d62728'}
LABELS = {'math': 'Math knot-positive pilot', 'coding': 'Coding knot-positive pilot'}


def load_gap(df: pd.DataFrame, feature: str, domain: str):
    sub = df[(df['feature'] == feature) & (df['domain'] == domain)].copy()
    correct = sub[sub['is_correct'] == 1].set_index('anchor_pct')['mean_value']
    incorrect = sub[sub['is_correct'] == 0].set_index('anchor_pct')['mean_value']
    return [(anchor, float(correct.loc[anchor] - incorrect.loc[anchor])) for anchor in ANCHORS]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROFILE_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    panel_specs = [
        ('traj_reflection_count', 'Correct − incorrect reflection count', 'Positive = correct runs retain fewer late reflections'),
        ('tok_conf_prefix', 'Correct − incorrect prefix confidence', 'Positive = correct runs remain more confident'),
    ]

    for ax, (feature, title, subtitle) in zip(axes, panel_specs):
        for domain in ['math', 'coding']:
            series = load_gap(df, feature, domain)
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(xs, ys, marker='o', linewidth=2.2, markersize=6, color=COLORS[domain], label=LABELS[domain])
        ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.set_xticks(ANCHORS)
        ax.set_xlabel('Trace anchor (%)')
        ax.set_title(title, fontsize=11)
        ax.text(0.02, 0.03, subtitle, transform=ax.transAxes, fontsize=8.5, color='dimgray')
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel('Gap value')
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')

    fig.suptitle('Knot semantics differ across math and coding', fontsize=13, y=1.03)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'Saved {OUT_PNG}')
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
