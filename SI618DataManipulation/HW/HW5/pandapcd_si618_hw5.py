# Calculate the total stars for each business category
# Original by Dr. Yuhang Wang and Josh Gardner
# Updated by Danaja Maldeniya
"""
To run on Cavium cluster:
spark-submit --master yarn --queue umsi618f21 --num-executors 16 --executor-memory 1g --executor-cores 2 spark_avg_stars_per_category.py

To get results:
hadoop fs -getmerge total_reviews_per_category_output total_reviews_per_category_output.txt
"""

import json
from pyspark import SparkContext

sc = SparkContext(appName="PySparksi618f19_total_reviews_per_category")

input_file = sc.textFile("/var/umsi618f21/hw5/yelp_academic_dataset_business.json")


def mapper(line):
    ans = list()
    line = json.loads(line)
    city = line.get("city", None)
    try:
        categories = line["categories"].split(", ")
    except Exception:
        categories = ["Unknown"]
    num = 1
    review_count = line["review_count"]
    rating = line["stars"] 
    try:
        assert line["attributes"]["WheelchairAccessible"] == "True"
        wheel = 1
    except Exception:
        wheel = 0
    parking = 0
    for key in ["garage", "street", "lot"]:
        try:
            v = eval(line["attributes"]["BusinessParking"])[key]
            if v in ["True", True]: 
                parking = 1
                break
        except Exception:
            pass
    for category in categories:
        ans.append([(city, category), (parking, wheel, num, rating, review_count)])
    return ans


def reducer(x, y):
    return [
        x[0] + y[0],
        x[1] + y[1],
        x[2] + y[2],
        x[3] + y[3],
        x[4] + y[4]
    ]


def sorter(line):
    city, category = line[0]
    parking, wheel, num, rating, review_count = line[1]
    return [(city, -num, category), (rating / num, wheel, parking)]


def output(line):
    city, num, category = line[0]
    rating, wheel, parking = line[1]
    return "{}\t{}\t{}\t{}\t{}\t{}".format(city, category, -num, rating, wheel, parking)


cat_stars = (
    input_file.flatMap(mapper)
    .filter(lambda line: line[-1][-1] > 0)
    .reduceByKey(reducer)
    .map(sorter)
    .sortByKey()
    .map(output)
)

cat_stars.collect()
cat_stars.saveAsTextFile("total_reviews_per_category_output")
