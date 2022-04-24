from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
import pyspark.pandas as ps
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
def process_binance():
    '''clean the raw data from Binance API'''
    ret = dict()
    for name in ["BTCUSDT", "ETHUSDT"]:
        df = ps.read_csv("../data/dirty/{}.csv".format(name)).drop(["OpenTime", "CloseTime", "TradeValue", "TradeCount", "BuyValue", "TradeVolume"], axis=1)
        df["Dates"] = (ps.to_datetime(df["TimeStamp"], format="%Y%m%d%H%M%S%f") - ps.to_datetime("20180101")) // 86400
        ret[name] = df.dropna(how="any", axis=0).drop(["TimeStamp"])
    return ret

def process_index():
    '''clean the raw data from Yahoo Finance'''
    names = {"GSPC": "SP500", "IXIC": "NSDQ", "GCF": "GOLD", "TNX": "TB"}
    ret = dict()
    tnx = None
    for key, val in names.items():
        df = ps.read_csv("../data/dirty/{}.csv".format(key)).drop(["Adj Close", "Volume"], axis=1)
        df["Dates"] = (ps.to_datetime(df["Date"], format="%Y-%m-%d") - ps.to_datetime("20180101")) // 86400
        df = df.drop(["Date"], axis=1).dropna(how="any", axis=0)
        df = df[df["Open"] != "null"]
        if key != "TNX": 
            ret[val] = df
        else: tnx = df
    tnx["Open"], tnx["High"], tnx["Low"], tnx["Close"] = 1 / tnx["Open"].astype(float), 1 / tnx["Low"].astype(float), 1 / tnx["High"].astype(float), 1 / tnx["Close"].astype(float)
    ret["TNX"] = tnx
    return ret

def preprocess():
    dfs = dict()
    dfs["crypto"] = process_binance()
    dfs["index"] = process_index()
    for val in dfs.values():
        for key, df in val.items():
            schema = ""
            for column in df.columns:
                df[column] = df[column].astype(float)
                schema += "{} FLOAT,".format(column)
            val[key] = spark.createDataFrame(df.values.tolist(), schema=schema[:-1])
    return dfs

def save(dfs):
    for key, val in dfs.items():
        for name, df in val.items():
            path = "../data/clean/{}.csv".format(name)
            df.write.format("csv").option("header", "true").save(path)

if __name__ == "__main__":
    dfs = preprocess()
    save(dfs)