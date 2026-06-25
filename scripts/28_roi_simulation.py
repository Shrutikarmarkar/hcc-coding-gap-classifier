"""
Calculates the breakeven claim volume between paying per-note for a
frontier LLM API vs. hosting the fine-tuned SLM on a dedicated GPU
instance -- using our own real benchmark cost number and a real,
sourced AWS GPU hosting rate, not estimates.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Real numbers from our own final benchmark
CLAUDE_COST_PER_NOTE = 0.002148

# Real AWS pricing: g4dn.xlarge (NVIDIA T4), right-sized for a 3.8B
# quantized model at burstable/low-concurrency inference
GPU_HOURLY_COST = 0.526
SLM_ANNUAL_HOSTING_COST = GPU_HOURLY_COST * 24 * 365

breakeven_notes = SLM_ANNUAL_HOSTING_COST / CLAUDE_COST_PER_NOTE

print(f"SLM annual fixed hosting cost (g4dn.xlarge, 24/7): ${SLM_ANNUAL_HOSTING_COST:,.2f}")
print(f"Claude API cost per note: ${CLAUDE_COST_PER_NOTE:.6f}")
print(f"Breakeven volume: {breakeven_notes:,.0f} notes/year")
print()

# Compare against a real, sourced reference point: UHC's ACA marketplace
# segment alone processed 6.4M claims in 2024 (CMS Transparency data)
uhc_aca_volume = 6_400_000
claude_cost_at_uhc_volume = uhc_aca_volume * CLAUDE_COST_PER_NOTE
savings_at_uhc_volume = claude_cost_at_uhc_volume - SLM_ANNUAL_HOSTING_COST

print(f"At UHC's ACA marketplace volume alone ({uhc_aca_volume:,} claims/year):")
print(f"  Frontier API cost:  ${claude_cost_at_uhc_volume:,.2f}/year")
print(f"  Fine-tuned SLM cost: ${SLM_ANNUAL_HOSTING_COST:,.2f}/year")
print(f"  Annual savings:      ${savings_at_uhc_volume:,.2f}")
print(f"  That's {uhc_aca_volume/breakeven_notes:.1f}x past the breakeven point")

# Build a simple volume-vs-cost chart
volumes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000, 6_400_000, 10_000_000, 20_000_000]
claude_costs = [v * CLAUDE_COST_PER_NOTE for v in volumes]
slm_costs = [SLM_ANNUAL_HOSTING_COST for _ in volumes]

plt.figure(figsize=(9, 5.5))
plt.plot(volumes, claude_costs, marker="o", label="Frontier LLM API (pay-per-note)", color="#14213D")
plt.plot(volumes, slm_costs, marker="o", label="Fine-tuned SLM (fixed GPU hosting)", color="#999999")
plt.axvline(breakeven_notes, color="red", linestyle="--", linewidth=1, label=f"Breakeven: {breakeven_notes:,.0f} notes/yr")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Claims/notes processed per year")
plt.ylabel("Annual cost (USD)")
plt.title("Annual cost: frontier API vs. self-hosted fine-tuned SLM, by volume")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/roi_breakeven_chart.png", dpi=150)
print()
print("Saved chart to data/processed/roi_breakeven_chart.png")