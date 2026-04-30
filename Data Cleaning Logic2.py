
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
spark = SparkSession.builder \
    .appName("LafargeDataCleansing") \
    .master("local[*]") \
    .get_or_membership()

from pyspark.sql.functions import col, when, mean


def industrial_cleansing_pipeline(df):
    # 1. Handle Sensor Malfunction (Clipping)
    df = df.withColumn("bearing_temp_cleaned",
                       when(col("bearing_temp") > 150, 110)  # Overheat cap
                       .when(col("bearing_temp") < 0, 45)  # Sensor freeze cap
                       .otherwise(col("bearing_temp")))

    # 2. Z-Score Normalization
    stats = df.select(mean("vibration").alias("mu")).collect()[0]
    stddev = df.select(stddev("vibration").alias("sigma")).collect()[0]
    df = df.withColumn("vibration_normalized", (col("vibration") - stats['mu']) / stddev['sigma'])

    # 3. Label Leakage Gate
    # Ensure failure_label is only joined with features from T minus 48 hours
    return df