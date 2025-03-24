from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when,count, avg, max, min, stddev
from pyspark.sql.types import StringType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.streaming import StreamingContext

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Banking Fraud Detection") \
    .getOrCreate()

# Load data (CSV or Parquet format)
df = spark.read.parquet("transaction_data.parquet", header=True, inferSchema=True)

# Preprocessing the data: cleaning and filtering out invalid records
df = df.dropna()  # Drop rows with null values
df = df.filter(col("Transaction Amount") > 0)  # Filter out invalid transactions with zero or negative amounts

# Example of creating features for each customer:
customer_stats = df.groupBy("Customer ID") \
    .agg(
        count("Transaction ID").alias("transaction_count"),
        avg("Transaction Amount").alias("avg_transaction_amount"),
        max("Transaction Amount").alias("max_transaction_amount"),
        min("Transaction Amount").alias("min_transaction_amount"),
        stddev("Transaction Amount").alias("stddev_transaction_amount")
    )

customer_stats = customer_stats.fillna({
    "transaction_count": 0,
    "avg_transaction_amount": 0,
    "max_transaction_amount": 0,
    "min_transaction_amount": 0,
    "stddev_transaction_amount": 0
})

customer_stats.show(5)

# Example to create fraud_label column if it doesn't exist


# Create the features vector
assembler = VectorAssembler(inputCols=["transaction_count", "avg_transaction_amount",
                                       "max_transaction_amount", "min_transaction_amount",
                                       "stddev_transaction_amount"], outputCol="features", handleInvalid="skip")
df_features  = assembler.transform(customer_stats)
# Train-test split
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)

# Example to create fraud_label column if it doesn't exist
train_data = train_data.withColumn("fraud_label", when(col("avg_transaction_amount") > 100, 1).otherwise(0))

train_data = train_data.withColumn("fraud_label1", when(col("avg_transaction_amount") < 1000, 1).otherwise(0))

# Define the model with the correct label column
rf = RandomForestClassifier(featuresCol="features", labelCol="fraud_label")

# Fit the model
model = rf.fit(train_data)

# Evaluate the model
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="stddev_transaction_amount")
auc = evaluator.evaluate(predictions)

print(f"Model AUC: {auc}")

