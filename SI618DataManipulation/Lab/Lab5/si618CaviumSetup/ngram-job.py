# Import the necessary Spark library classes, as well as sys
from pyspark import SparkConf, SparkContext
import sys

# Ensure that an input and output are specified on the command line
if len(sys.argv) < 3:
    print('Usage: ' + sys.argv[0] + ' <in> <out>')
    sys.exit(1)

# Grab the input and output
input = sys.argv[1]
output = sys.argv[2]

# Create a configuration for this Spark job
conf = SparkConf().setAppName('AnnualWordLength').set("spark.hadoop.validateOutputSpecs", "false")

# Create a context for the job. The context is used to manage the job at a
# high level.
sc = SparkContext(conf=conf)

# Read in the dataset and immediately transform all the lines in arrays
data = sc.textFile(input).map(lambda line: line.split('\t'))

# Create the 'length' dataset as mentioned above. This is done using the next
# two variables, and the 'length' dataset ends up in 'yearlyLength'
yearlyLengthAll = data.map(
    lambda arr: (int(arr[1]), float(len(arr[0])) * float(arr[2]))
)
yearlyLength = yearlyLengthAll.reduceByKey(lambda a, b: a + b)

# Create the 'words' dataset as mentioned above.
yearlyCount = data.map(
    lambda arr: (int(arr[1]), float(arr[2]))
).reduceByKey(
    lambda a, b: a + b
)

# Create the 'average_length' dataset as mentioned above.
yearlyAvg = yearlyLength.join(yearlyCount).map(
    lambda tup: (tup[0], tup[1][0] / tup[1][1])
)

# Save the results in the specified output directory.
yearlyAvg.saveAsTextFile(output)

# Finally, let Spark know that the job is done.
sc.stop()
