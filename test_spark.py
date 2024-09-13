from pyspark.sql import SparkSession

# Initialize a SparkSession
spark = SparkSession.builder \
    .appName("Filter Parquet and Save to S3") \
    .getOrCreate()

# Set AWS credentials (if not set globally or through IAM roles)
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "YOUR_ACCESS_KEY")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "YOUR_SECRET_KEY")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")

# Read Parquet files from S3
input_path = "s3a://your-input-bucket/path/to/parquet/files/"
df = spark.read.parquet(input_path)

# Filter the DataFrame by a specific value
# Replace 'column_name' with the name of the column you want to filter by
# Replace 'value_to_filter' with the value you want to filter
filtered_df = df.filter(df['column_name'] == 'value_to_filter')

# Save the filtered DataFrame to another S3 directory
output_path = "s3a://your-output-bucket/path/to/save/filtered/files/"
filtered_df.write.mode('overwrite').parquet(output_path)

# Stop the SparkSession
spark.stop()
