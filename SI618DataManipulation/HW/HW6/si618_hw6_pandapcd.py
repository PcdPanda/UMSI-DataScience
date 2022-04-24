from pyspark import SparkContext
from pyspark.sql import SQLContext
 
sqlContext = SQLContext(sc)
business_path = "/var/umsi618f21/hw6/yelp_academic_dataset_business.json"
review_path = "/var/umsi618f21/hw6/yelp_academic_dataset_review.json"
df_business = sqlContext.read.json(business_path)
df_review = sqlContext.read.json(review_path)
df_business.registerTempTable("df_business")
df_review.registerTempTable("df_review")
# Q1
sql = "SELECT COALESCE((stars - tmp.avg) / NANVL(tmp.std, 0), 0) as norm_rating, business_id, useful FROM df_review JOIN (SELECT AVG(stars) as avg, STD(stars) as std, user_id FROM df_review GROUP BY user_id) AS tmp ON df_review.user_id=tmp.user_id"
df_norm_user = sqlContext.sql(sql)
df_norm_user.registerTempTable("df_norm_user")
sqlContext.cacheTable("df_norm_user")
sql = "SELECT business_id, AVG(norm_rating) as business_rating FROM df_norm_user GROUP BY business_id ORDER BY business_rating DESC LIMIT 100"
df_p1 = sqlContext.sql(sql)
df_p1.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_hw6_output_1.tsv")

# Q2
sql = "SELECT business_id, AVG(norm_rating) as business_rating FROM df_norm_user GROUP BY business_id"
df_norm_business = sqlContext.sql(sql)
df_norm_business.registerTempTable("df_norm_business")
sql = "SELECT city, AVG(business_rating) as city_rating from df_business JOIN df_norm_business ON df_business.business_id=df_norm_business.business_id GROUP BY city ORDER BY city_rating DESC, city DESC"
df_p2 = sqlContext.sql(sql)
df_p2.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_hw6_output_2.tsv")

# Q3
sql = "SELECT business_id, AVG(norm_rating) as business_rating FROM df_norm_user WHERE useful > 0 GROUP BY business_id"
df_norm_business = sqlContext.sql(sql)
df_norm_business.registerTempTable("df_norm_business_useful")
sql = "SELECT city, AVG(business_rating) as city_rating from df_business JOIN df_norm_business_useful ON df_business.business_id=df_norm_business_useful.business_id GROUP BY city ORDER BY city_rating DESC, city DESC"
df_p3 = sqlContext.sql(sql)
df_p3.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_hw6_output_3.tsv") 





