"""Generate per-turn token comparison figure: Incremental (A) vs Full Recompute (D)."""

import os

import matplotlib.pyplot as plt
import numpy as np

# Data
turns = np.arange(1, 6)
tokens_a = [7850, 7933, 4667, 12905, 4637]
tokens_d = [36980, 5341, 73307, 13794, 106653]

cum_a = np.cumsum(tokens_a)
cum_d = np.cumsum(tokens_d)

# Print summary table
print("Turn | A (Incr.) | D (Full)  | A Cumul.  | D Cumul.  | Per-Turn Savings")
print("-----+-----------+-----------+-----------+-----------+-----------------")
for i in range(5):
    saving = (1 - tokens_a[i] / tokens_d[i]) * 100 if tokens_d[i] else 0
    print(
        f"  {i+1}  | {tokens_a[i]:>9,} | {tokens_d[i]:>9,} "
        f"| {cum_a[i]:>9,} | {cum_d[i]:>9,} | {saving:>14.1f}%"
    )

total_saving = (1 - cum_a[-1] / cum_d[-1]) * 100
print(f"\nTotal input tokens — A: {cum_a[-1]:,}  D: {cum_d[-1]:,}")
print(f"Total savings: {total_saving:.1f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: per-turn bar chart
width = 0.35
ax1.bar(turns - width / 2, tokens_a, width, color="#4C72B0", label="Incremental (A)")
ax1.bar(turns + width / 2, tokens_d, width, color="#C44E52", label="Full Recompute (D)")
for i, (va, vd) in enumerate(zip(tokens_a, tokens_d)):
    ax1.text(turns[i] - width / 2, va + 1500, f"{va:,}", ha="center", va="bottom", fontsize=7)
    ax1.text(turns[i] + width / 2, vd + 1500, f"{vd:,}", ha="center", va="bottom", fontsize=7)
ax1.set_xlabel("Turn")
ax1.set_ylabel("Input Tokens")
ax1.set_title("Per-Turn Input Token Usage")
ax1.set_xticks(turns)
ax1.legend()
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right: cumulative line chart
ax2.plot(turns, cum_a, "o-", color="#4C72B0", label="Incremental (A)", linewidth=2, markersize=6)
ax2.plot(turns, cum_d, "o-", color="#C44E52", label="Full Recompute (D)", linewidth=2, markersize=6)
ax2.annotate(
    "84% total savings",
    xy=(5, cum_a[-1]),
    xytext=(3.5, (cum_a[-1] + cum_d[-1]) / 2),
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black"),
    ha="center",
)
ax2.set_xlabel("Turn")
ax2.set_ylabel("Cumulative Input Tokens")
ax2.set_title("Cumulative Token Growth: O(k) vs O(k\u00b2)")
ax2.set_xticks(turns)
ax2.legend()
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "streaming", "per_turn_token_comparison.png"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved to {os.path.abspath(out_path)}")
