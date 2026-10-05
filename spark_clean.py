"""
spark_clean.py — NYC Yellow Taxi Pipeline
Framework: PySpark 3.5.1
Cluster: spark://127.0.0.1:7077
"""
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

T0 = time.perf_counter()

# -------- 0. Session --------
spark = SparkSession.builder \
    .appName("TaxiCleaning_Spark") \
    .master("spark://127.0.0.1:7077") \
    .config("spark.executor.memory", "500m") \
    .config("spark.driver.memory", "500m") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"[{time.perf_counter()-T0:.2f}s] Session started")

# -------- 1. Ingest --------
trips = spark.read.parquet("/home/vishnu/taxi_data/raw/yellow_tripdata_2024-01.parquet")
zones = spark.read.option("header", True).csv("/home/vishnu/taxi_data/raw/taxi_zone_lookup.csv")
print(f"[{time.perf_counter()-T0:.2f}s] Ingested: {trips.count()} trips, {zones.count()} zones")

# -------- 2. Cleanse --------
trips_clean = trips \
    .dropDuplicates(["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID"]) \
    .dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime","trip_distance","PULocationID","fare_amount"]) \
    .filter(F.col("trip_distance") > 0) \
    .filter(F.col("fare_amount") >= 0)

trips_clean = trips_clean \
    .withColumn("pickup_ts",  F.to_timestamp("tpep_pickup_datetime")) \
    .withColumn("dropoff_ts", F.to_timestamp("tpep_dropoff_datetime"))

print(f"[{time.perf_counter()-T0:.2f}s] Cleansed: {trips_clean.count()} trips")

# -------- 3. Heavy join --------
zones_small = zones.select(
    F.col("LocationID").cast("int").alias("zone_id"),
    F.col("Borough"),
    F.col("Zone")
)

joined = trips_clean.join(zones_small, trips_clean.PULocationID == zones_small.zone_id, "inner")
print(f"[{time.perf_counter()-T0:.2f}s] Joined: {joined.count()} rows")

# -------- 3b. UDF (speed per hour) --------
@F.udf(returnType=DoubleType())
def avg_speed_udf(distance, pickup, dropoff):
    if distance is None or pickup is None or dropoff is None:
        return None
    seconds = (dropoff - pickup).total_seconds()
    if seconds <= 0:
        return None
    return float(distance) / (seconds / 3600.0)

t_udf_start = time.perf_counter()

with_speed = joined.withColumn(
    "speed_mph",
    avg_speed_udf(F.col("trip_distance"), F.col("pickup_ts"), F.col("dropoff_ts"))
).withColumn("hour_of_day", F.hour("pickup_ts"))

result = with_speed.groupBy("hour_of_day", "Borough") \
    .agg(
        F.avg("speed_mph").alias("avg_speed_mph"),
        F.count("*").alias("trip_count")
    ).orderBy("hour_of_day", "Borough")

row_count = result.count()
udf_time = time.perf_counter() - t_udf_start
print(f"[{time.perf_counter()-T0:.2f}s] UDF+group done in {udf_time:.2f}s — {row_count} result rows")

# -------- 4. Export --------
result.write.mode("overwrite").parquet("/home/vishnu/taxi_data/processed/spark_output")
print(f"[{time.perf_counter()-T0:.2f}s] Exported parquet")

TOTAL = time.perf_counter() - T0
print(f"\n=== SPARK TOTAL TIME: {TOTAL:.2f}s ===")
print(f"=== SPARK UDF TIME:   {udf_time:.2f}s ===")

spark.stop()
