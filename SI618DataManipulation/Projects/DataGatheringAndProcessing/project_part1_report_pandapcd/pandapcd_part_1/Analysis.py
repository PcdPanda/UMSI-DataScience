from pyspark import SparkContext, pandas
from pyspark.sql import SQLContext, SparkSession
from BackTest import load
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext(sc)

def load_ret():
    '''load the dataset'''
    ret = dict()
    datatype = {"crypto":["BTCUSDT", "ETHUSDT"], "index":["GOLD", "NSDQ", "SP500", "TNX"]}
    for strategy in ["auto", "macd", "boll"]:
        ret[strategy] = dict()
        for asset_type, val in datatype.items():
            if asset_type == "index" and strategy == "boll": continue
            ret[strategy][asset_type] = dict()
            for ticker in val:
                path = "../result/{}_{}_{}.csv".format(strategy, asset_type, ticker)
                ret[strategy][asset_type][ticker] = {"Revenue": sqlContext.read.option("header", "true").csv(path, schema="Close FLOAT, Revenue FLOAT, Dates FLOAT")}
                if strategy != "auto":
                    path =  "../result/{}_{}_{}_record.csv".format(strategy, asset_type, ticker)
                    ret[strategy][asset_type][ticker]["Record"] = sqlContext.read.option("header", "true").csv(path, schema="TradeRevenue FLOAT, BuyDate FLOAT,SellDate FLOAT")
    return ret


def analyze_price(dfs):
    '''analyze the performance of price'''
    price = dict()
    sql = "SELECT {asset}_df.Dates as Dates, Close / start as {asset} FROM {asset}_df CROSS JOIN (SELECT Close as start FROM {asset}_df ORDER BY Dates LIMIT 1) AS tmp"
    for asset in ["ETHUSDT", "BTCUSDT"]:
        dfs["crypto"][asset].createOrReplaceTempView("{}_df".format(asset))
        price[asset] = sqlContext.sql(sql.format(asset=asset)).dropDuplicates(["Dates"])
    ans = price["ETHUSDT"].join(price["BTCUSDT"], ["Dates"], how="left")
    for asset in ["SP500", "TNX", "NSDQ", "GOLD"]:
        dfs["index"][asset].createOrReplaceTempView("{}_df".format(asset))
        price[asset] = sqlContext.sql(sql.format(asset=asset))
        ans = ans.join(price[asset], ["Dates"], how="left")
    ans.createOrReplaceTempView("price")
    return ans

def analyze_strategy(ret):
    '''analyze the performance of strategy'''
    ans = None
    for strategy, strategy_data in ret.items():
        for asset_type, asset_df in strategy_data.items():
            for ticker, df in asset_df.items():
                df_name = "{}_{}_df".format(strategy, ticker)
                df["Revenue"].createOrReplaceTempView(df_name)
                sql = "SELECT Dates, Revenue as {strategy}_{ticker} FROM {df_name}".format(**locals())
                df = sqlContext.sql(sql).dropDuplicates(["Dates"])
                ans = df if ans == None else ans.join(df.dropDuplicates(["Dates"]), ["Dates"], how="left")
    return ans

def analyze(dfs, ret):
    price = analyze_price(dfs)
    corr, predict = list(), list()
    w = Window().partitionBy().orderBy(col("Dates"))
    for asset1 in list(price.columns):
        if asset1 != "Dates":
            row = [asset1]
            row_predict = [asset1]
            predict_price = price.withColumn("predict", lag(asset1, 1).over(w))
            for asset2 in list(price.columns):
                if asset2 != "Dates":
                    row.append(price.corr(asset1, asset2))
                    row_predict.append(predict_price.corr("predict", asset2))
            predict.append(row_predict)
            corr.append(row)
    schema = 'index String, ' + ' FLOAT, '.join(price.columns[1:]) + ' FLOAT'
    corr = spark.createDataFrame(corr, schema).to_pandas_on_spark()      
    predict = spark.createDataFrame(predict, schema).to_pandas_on_spark()      
    strategy = analyze_strategy(ret)
    ans = strategy.join(price, ["Dates"]).orderBy(["Dates"], how="left")
    ans = ans.to_pandas_on_spark().fillna(method="ffill").iloc[1:]
    res, max_drawdown, std, index, records = list(), list(), list(), list(), list()
    for column in ans.columns:
        row = list()
        if column != "Dates":
            row.append(column)
            row.append(ans[column].tolist()[-1] / ans[column].tolist()[0])
            drawdown = 1 - ans[column] / ans[column].cummax()
            row.append(drawdown.max())
            row.append(ans[column].std())
            res.append(row)
    res = spark.createDataFrame(res, schema="index String, ret FLOAT, max_drawdown FLOAT, std FLOAT").to_pandas_on_spark()
    for strategy in ["macd", "boll"]:
        for asset_type, asset_df in ret[strategy].items():
            for ticker in asset_df.keys():
                df = asset_df[ticker]["Record"].to_pandas_on_spark()
                if "USDT" in ticker: ticker = ticker.replace("USDT", "")
                row = ["{}_{}".format(strategy, ticker), len(df), len(df[df["TradeRevenue"] >0]) / len(df), df["TradeRevenue"].mean()]
                records.append(row)
    records = spark.createDataFrame(records, schema="index String, trade_num INT, win_rate FLOAT, avg_revenue FLOAT").to_pandas_on_spark()           
    return ans, res, records, corr, predict

def save(ans, res, records, corr, predict):
    ans_path = "../result/ans.csv"
    ans.to_csv(ans_path)
    res_path = "../result/res.csv"
    res.to_csv(res_path)
    records_path = "../result/records.csv"
    records.to_csv(records_path)
    corr_path = "../result/corr.csv"
    corr.to_csv(corr_path)
    predict_path = "../result/predict.csv"
    predict.to_csv(predict_path)


if __name__ == "__main__":
    dfs = load()
    ret = load_ret()
    save(*analyze(dfs, ret))