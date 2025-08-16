from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, round as spark_round

USERS_CSV = "users.csv"
PURCHASES_CSV = "purchases.csv"
PRODUCTS_CSV = "products.csv"

spark = SparkSession.builder.appName("Purchases").getOrCreate()


# 1. Завантажте та прочитайте кожен CSV-файл як окремий DataFrame.
users = spark.read.options(**{"header": True, "inferSchema": True}).csv(USERS_CSV)
purchases = spark.read.options(**{"header": True, "inferSchema": True}).csv(PURCHASES_CSV)
products = spark.read.options(**{"header": True, "inferSchema": True}).csv(PRODUCTS_CSV)

# 2. Очистіть дані, видаляючи будь-які рядки з пропущеними значеннями.
users_clean = users.na.drop()
purchases_clean = purchases.na.drop()
products_clean = products.na.drop()

purchases_clean = purchases_clean.withColumn("quantity", col("quantity").cast("double"))
products_clean = products_clean.withColumn("price", col("price").cast("double"))

# 3. Визначте загальну суму покупок за кожною категорією продуктів.
purchases_with_products = purchases_clean.join(products_clean, on="product_id", how="inner")
purchases_with_products = purchases_with_products.withColumn("purchase_total", col('quantity') * col("price"))

total_by_category = purchases_with_products.groupBy("category").agg(
    spark_round(spark_sum("purchase_total"), 2).alias("total_spent")
).orderBy(col("total_spent").desc())

total_by_category.show()

# 4. Визначте суму покупок за кожною категорією продуктів для вікової категорії від 18 до 25 включно.
purchases_with_users = purchases_clean.join(users_clean, on="user_id", how="inner")
purchases_in_age_range = purchases_with_users.filter((col("age") >= 18) & (col("age") <= 25))
purchases_in_age_range = purchases_in_age_range.join(products_clean, on="product_id", how="inner")
purchases_in_age_range = purchases_in_age_range.withColumn("purchase_total", col("quantity") * col("price"))

total_in_age_range_by_category = purchases_in_age_range.groupBy("category").agg(
    spark_round(spark_sum("purchase_total"), 2).alias("total_spent_18_25")
).orderBy(col("total_spent_18_25").desc())

total_in_age_range_by_category.show()

# 5. Визначте частку покупок за кожною категорією товарів від сумарних витрат для вікової категорії від 18 до 25 років.
total_age_range_row = purchases_in_age_range.agg(spark_sum("purchase_total").alias("total_18_25")).collect()[0]
total_age_range = total_age_range_row["total_18_25"] if total_age_range_row["total_18_25"] is not None else 0.0

if total_age_range == 0.0:
    shares_df = total_in_age_range_by_category.withColumn("percentage", spark_round(col("total_spent_18_25") * 0, 2))
else:
    shares_df = total_in_age_range_by_category.withColumn(
        "percentage", 
        spark_round((col("total_spent_18_25") / total_age_range) * 100, 2)
    )

shares_df.show()

# 6. Виберіть 3 категорії продуктів з найвищим відсотком витрат споживачами віком від 18 до 25 років.
top_three_categories = shares_df.orderBy(col("percentage").desc()).limit(3)
top_three_categories.show()

spark.stop()
