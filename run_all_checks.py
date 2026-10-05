"""
run_all_checks.py — Combined verification + report data extraction
Run this AFTER both pipelines have completed.
"""
import pandas as pd
import os
from pathlib import Path

OUT = Path("/media/sf_vmshare/results")
OUT.mkdir(exist_ok=True)

print("=" * 60)
print("PIPELINE PARITY CHECK + REPORT DATA EXTRACTION")
print("=" * 60)

# -------- Load both outputs --------
s = pd.read_parquet("/home/vishnu/taxi_data/processed/spark_output/")
r = pd.read_parquet("/home/vishnu/taxi_data/processed/ray_output.parquet")

# Ray sometimes has a stray 'index' column
if "index" in r.columns:
    r = r.drop(columns=["index"])

# Normalize column order & dtypes
s = s[sorted(s.columns)]
r = r[sorted(r.columns)]

print(f"\nSpark rows: {len(s)} | Ray rows: {len(r)}")
print(f"Spark cols: {sorted(s.columns.tolist())}")
print(f"Ray cols:   {sorted(r.columns.tolist())}")

# -------- Join on the composite key --------
merged = s.merge(r, on=["hour_of_day", "Borough"],
                 how="outer", suffixes=("_spark", "_ray"))

# Handle rows present in only one framework
merged["diff"] = (merged["avg_speed_mph_spark"] - merged["avg_speed_mph_ray"]).abs()

max_diff = merged["diff"].max()
mean_diff = merged["diff"].mean()
n_rows = len(merged)

print(f"\nRows compared (outer join): {n_rows}")
print(f"Max |avg_speed difference|: {max_diff}")
print(f"Mean |avg_speed difference|: {mean_diff}")

# Parity verdict
PARITY_TOL = 0.01  # mph
if max_diff < PARITY_TOL:
    print(f"\n>>> PARITY VERIFIED (max diff {max_diff:.6f} < {PARITY_TOL})")
else:
    print(f"\n>>> PARITY WARNING: max diff {max_diff:.6f} exceeds tolerance {PARITY_TOL}")
    # Show top 10 worst rows
    worst = merged.nlargest(10, "diff")[["hour_of_day","Borough",
                                          "avg_speed_mph_spark",
                                          "avg_speed_mph_ray","diff"]]
    print(worst.to_string())

# -------- Per-hour summary for report --------
print("\n" + "=" * 60)
print("AGGREGATED COMPARISON (Spark vs Ray)")
print("=" * 60)

hourly = merged.groupby("hour_of_day").agg(
    spark_speed=("avg_speed_mph_spark", "mean"),
    ray_speed=("avg_speed_mph_ray", "mean"),
    spark_count=("trip_count_spark", "sum"),
    ray_count=("trip_count_ray", "sum"),
).round(4)

print(hourly.to_string())

# -------- Save everything for report --------
merged.to_csv(OUT / "parity_full.csv", index=False)
hourly.to_csv(OUT / "parity_hourly.csv")

with open(OUT / "parity_summary.txt", "w") as f:
    f.write(f"Spark rows: {len(s)}\n")
    f.write(f"Ray rows:   {len(r)}\n")
    f.write(f"Max |diff|: {max_diff}\n")
    f.write(f"Mean |diff|: {mean_diff}\n")
    f.write(f"Parity verified: {max_diff < PARITY_TOL}\n")

print(f"\nSaved to {OUT}/:")
for p in OUT.iterdir():
    print("  ", p.name)

print("\nDONE.")