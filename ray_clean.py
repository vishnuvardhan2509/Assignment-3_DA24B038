"""
ray_clean.py — NYC Yellow Taxi Pipeline
Framework: Ray 2.59.0 + Ray Data
Cluster: local Ray head (2 CPUs)
"""
import time
import ray
import pandas as pd

T0 = time.perf_counter()

# -------- 0. Init --------
ray.init(address="auto", ignore_reinit_error=True)
print(f"[{time.perf_counter()-T0:.2f}s] Ray initialized")

# -------- 1. Ingest --------
ds = ray.data.read_parquet("/home/vishnu/taxi_data/raw/yellow_tripdata_2024-01.parquet")
zones = pd.read_csv("/home/vishnu/taxi_data/raw/taxi_zone_lookup.csv")
zones["LocationID"] = zones["LocationID"].astype("int64")
print(f"[{time.perf_counter()-T0:.2f}s] Ingested: {ds.count()} trips, {len(zones)} zones")

# -------- 2. Cleanse --------
def clean(batch: pd.DataFrame) -> pd.DataFrame:
    batch = batch.drop_duplicates(subset=["VendorID","tpep_pickup_datetime",
                                         "tpep_dropoff_datetime","PULocationID"])
    batch = batch.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime",
                                 "trip_distance","PULocationID","fare_amount"])
    batch = batch[batch["trip_distance"] > 0]
    batch = batch[batch["fare_amount"] >= 0]
    batch["pickup_ts"]  = pd.to_datetime(batch["tpep_pickup_datetime"])
    batch["dropoff_ts"] = pd.to_datetime(batch["tpep_dropoff_datetime"])
    return batch

ds_clean = ds.map_batches(clean, batch_format="pandas")
print(f"[{time.perf_counter()-T0:.2f}s] Cleansed: {ds_clean.count()} trips")

# -------- 3. Heavy join --------
zone_lookup = {int(r.LocationID): r.Borough for r in zones.itertuples()}

def join_zone(batch: pd.DataFrame) -> pd.DataFrame:
    batch["Borough"] = batch["PULocationID"].map(
        lambda x: zone_lookup.get(int(x), "Unknown")
    )
    return batch

ds_joined = ds_clean.map_batches(join_zone, batch_format="pandas")
print(f"[{time.perf_counter()-T0:.2f}s] Joined: {ds_joined.count()} rows")

# -------- 3b. UDF — speed per hour --------
t_udf_start = time.perf_counter()

def compute_speed(batch: pd.DataFrame) -> pd.DataFrame:
    seconds = (batch["dropoff_ts"] - batch["pickup_ts"]).dt.total_seconds()
    batch["speed_mph"] = batch["trip_distance"] / (seconds / 3600.0)
    batch.loc[(seconds <= 0) | batch["speed_mph"].isna(), "speed_mph"] = None
    batch["hour_of_day"] = batch["pickup_ts"].dt.hour
    return batch[["hour_of_day","Borough","speed_mph"]]

ds_speed = ds_joined.map_batches(compute_speed, batch_format="pandas")

result = ds_speed.groupby(["hour_of_day","Borough"]).mean("speed_mph").to_pandas()
trip_counts = ds_speed.groupby(["hour_of_day","Borough"]).count().to_pandas()
result = result.rename(columns={"mean(speed_mph)": "avg_speed_mph"})
result["trip_count"] = trip_counts["count()"].values
result = result.reset_index()

udf_time = time.perf_counter() - t_udf_start
print(f"[{time.perf_counter()-T0:.2f}s] UDF+group done in {udf_time:.2f}s — {len(result)} result rows")

# -------- 4. Export --------
result.to_parquet("/home/vishnu/taxi_data/processed/ray_output.parquet")
print(f"[{time.perf_counter()-T0:.2f}s] Exported parquet")

TOTAL = time.perf_counter() - T0
print(f"\n=== RAY TOTAL TIME: {TOTAL:.2f}s ===")
print(f"=== RAY UDF TIME:   {udf_time:.2f}s ===")

ray.shutdown()