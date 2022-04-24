PORT="$(shuf -i 10000-60000 -n 1)"
spark-submit --master yarn --queue umsi618f21 --num-executors 2 --executor-memory 4g --executor-cores 2  --conf spark.hadoop.validateOutputSpecs=false $1 $2 $3 spark.ui.port="${PORT}"
