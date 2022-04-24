from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext(appName="PySparksi618f21lab6")
sqlContext = SQLContext(sc)
df = sqlContext.read.csv("/var/umsi618f21/lab6/na_ranked_team.csv", header=True)
df.registerTempTable("LOL_Match")
# TODO 精度问题
# Q1
sql = "Select summonerId, COUNT(*) as matches, SUM((kill + assist)/(death + 1))/ COUNT(*) as kda FROM LOL_Match GROUP BY summonerId HAVING matches>=10 ORDER BY kda DESC"
p1_df = sqlContext.sql(sql)
p1_df.collect()
p1_df.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_lab6_output_1.tsv")

# Q2
sql = "Select matchId, winner, SUM((kill + assist) / (death + 1)) / 5 as avg_kda FROM LOL_Match GROUP BY matchId, winner"
p2_avg = sqlContext.sql(sql)
p2_avg.registerTempTable("Team_Avg_Kda")
sql = "Select summonerId, (kill + assist)/(death + 1) / (avg_kda + 1) as normalized_kda FROM LOL_Match JOIN Team_Avg_Kda WHERE LOL_Match.matchId=Team_Avg_Kda.matchId AND LOL_Match.winner=Team_Avg_Kda.winner"
p2_kda = sqlContext.sql(sql)
p2_kda.registerTempTable("p2_kda")
sql = "Select summonerId, COUNT(*) as matches, SUM(normalized_kda)/ COUNT(*) as kda FROM p2_kda GROUP BY summonerId HAVING matches>=10 ORDER BY kda DESC"
p2_df = sqlContext.sql(sql)
p2_df.take(5)
p2_df.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_lab6_output_2.tsv")

# Q3
sql = "SELECT a.championName as a_champion, b.championName as b_champion, a.predictedRole, COUNT(*) as matches, AVG(((a.kill + a.assist) / (a.death + 1)) / ((b.kill + b.assist) / (b.death + 1))) FROM LOL_Match a JOIN LOL_Match b ON a.predictedRole=b.predictedRole and a.matchId=b.matchId WHERE a.championName < b.championName AND a.winner != b.winner GROUP BY a.predictedRole, a.championName, b.championName HAVING matches >= 10 ORDER BY a.championName, a.predictedRole, matches DESC, b.championName"
p3_df = sqlContext.sql(sql)
p3_df.write.format("csv").option("header", "false").option("sep", "\t").save("./pandapcd_si618_lab6_output_3.tsv")
