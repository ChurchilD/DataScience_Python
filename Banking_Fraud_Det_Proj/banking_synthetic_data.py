from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import random
from datetime import datetime, timedelta

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Generate Synthetic Transaction Data") \
    .getOrCreate()


# Function to generate synthetic data
def generate_synthetic_data(num_rows):
    transaction_data = []
    transaction_types = ["withdrawal", "deposit", "transfer"]
    merchant_categories = ["electronics", "clothing", "grocery", "restaurant", "healthcare"]

    for i in range(num_rows):
        transaction_id = f"TXN{i + 1}"
        customer_id = random.randint(1, 1000)
        transaction_amount = round(random.uniform(10, 5000), 2)
        transaction_type = random.choice(transaction_types)
        timestamp = datetime.now() - timedelta(days=random.randint(1, 30))
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        merchant_category = random.choice(merchant_categories)
        location = (round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6))  # Latitude, Longitude
        previous_transactions = random.randint(0, 100)

        transaction_data.append((transaction_id, customer_id, transaction_amount, transaction_type, timestamp_str,
                                 merchant_category, location, previous_transactions))

    return transaction_data


# Generate synthetic transaction data (e.g., 1000 rows)
num_rows = 1000
data = generate_synthetic_data(num_rows)

# Define schema for the DataFrame
schema = ["Transaction ID", "Customer ID", "Transaction Amount", "Transaction Type", "Timestamp",
          "Merchant Category", "Location", "Previous Transactions"]

# Create a PySpark DataFrame
df = spark.createDataFrame(data, schema)

# Show a few rows of the generated DataFrame
df.show(5)

# Save the DataFrame as a Parquet file
df.write.parquet("transaction_data.parquet", mode="overwrite")

print("Parquet file saved successfully!")