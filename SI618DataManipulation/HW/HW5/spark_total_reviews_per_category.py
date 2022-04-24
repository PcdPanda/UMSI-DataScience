# Calculate the total stars for each business category
# Original by Dr. Yuhang Wang and Josh Gardner
# Updated by Danaja Maldeniya
'''
To run on Cavium cluster:
spark-submit --master yarn --queue umsi618f21 --num-executors 16 --executor-memory 1g --executor-cores 2 spark_avg_stars_per_category.py

To get results:
hadoop fs -getmerge total_reviews_per_category_output total_reviews_per_category_output.txt
'''

import json
from pyspark import SparkContext
sc = SparkContext(appName="PySparksi618f19_total_reviews_per_category")

input_file = sc.textFile("/var/umsi618f21/hw5/yelp_academic_dataset_business.json")

def cat_reviews(data):
    cat_review_list = []
    reviews = data.get('review_count', None)
    categories_raw = data.get('categories', None)
    if categories_raw:
        categories = categories_raw.split(', ')
        for c in categories:
            if reviews != None:
                cat_review_list.append((c, reviews))
    return cat_review_list


cat_stars = input_file.map(lambda line: json.loads(line)) \
                      .flatMap(cat_reviews) \
                      .reduceByKey(lambda x, y: x + y)

cat_stars.collect()
cat_stars.saveAsTextFile("total_reviews_per_category_output")
