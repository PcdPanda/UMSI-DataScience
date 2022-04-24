from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
import pyspark.pandas as ps
from typing import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
ps.set_option('compute.ops_on_diff_frames', True)
sqlContext = SQLContext(sc)

def load():
    '''Read the csv clean data'''
    dfs = {"crypto": dict(), "index": dict()}
    for ticker in ["ETHUSDT", "BTCUSDT"]:
        path = "../data/clean/{}.csv".format(ticker)
        df = sqlContext.read.option("header", "true").csv(path, schema="Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, BuyVolume FLOAT, Dates FLOAT").orderBy(["Dates"])
        dfs["crypto"][ticker] = df
    for ticker in ["GOLD", "NSDQ", "SP500", "TNX"]:
        path = "../data/clean/{}.csv".format(ticker)
        df = sqlContext.read.option("header", "true").csv(path, schema="Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Dates FLOAT").orderBy(["Dates"])
        dfs["index"][ticker] = df
    return dfs

def simulate(dfs: dict, strategy: Callable):
    '''A Spark framework for simulation'''
    ret = {"crypto": dict(), "index": dict()}
    for asset_type in ["crypto", "index"]:
        for ticker, df in dfs[asset_type].items():
            revenue, trade_record = strategy(df.to_pandas_on_spark(), asset_type) 
            # call the strategy
            if trade_record:
                trade_record = spark.createDataFrame(trade_record, schema="TradeRevenue FLOAT, BuyDate FLOAT, SellDate FLOAT")
                trade_record.createOrReplaceTempView("trade_record_{}".format(ticker))
            revenue = spark.createDataFrame(revenue, schema="Close FLOAT, Revenue FLOAT, Dates FLOAT")
            revenue.createOrReplaceTempView("Revenue_{}".format(ticker))
            ret[asset_type][ticker] = {"Revenue": revenue, "Record": trade_record}
    return ret


def auto(df, asset_type: str):
    '''automatic strategy'''
    signal = list()
    trades, num, buy_date, cash, revenue = 0, 0, 0, 1, 1
    for close, dates in df[["Close", "Dates"]].values.tolist():
        if dates >= buy_date: # trade time
            trades += 1
            cash -= 1 / 198
            num += 1 / (close * 198)
            buy_date += 7
        revenue = cash + num * close
        signal.append([close, revenue, dates])
    return signal, list()
    
def macd(df, asset_type: str):
    '''MACD strategy'''
    trades, num, buy_date, cash, revenue = 0, 0, 0, 1, 1
    ema_12, ema_26, dea = None, None, 0
    signal, trade_info, trade_record = list(), list(), list()
    for close, date in df[["Close", "Dates"]].values.tolist():
        if not ema_12 and not ema_26: ema_12, ema_26 = close, close
        last_dea = dea
        if asset_type == "index":
            ema_12 = ema_12 * 11 / 13 + close * 2 / 13
            ema_26 = ema_26 * 25 / 27 + close * 2 / 27
            dea = dea * 8 / 10 + (ema_12 - ema_26) * 2 / 10
        else:
            ema_12 = ema_12 * 76 / 78 + close * 2 / 78
            ema_26 = ema_26 * 160 / 162 + close * 2 / 162
            dea = dea * 58 / 60 + (ema_12 - ema_26) * 2 / 60
        if last_dea <= 0 and dea > 0 and cash: # buy
            num += cash / close
            cash = 0
            trade_info = [cash + num * close, date]
        elif last_dea >= 0 and dea < 0 and num:
            cash += num * close
            num = 0
            trade_info.append(date)
            trade_info[0] = cash + num * close - trade_info[0]
            trade_record.append(trade_info)
            trade_info = list()
        revenue = cash + num * close
        signal.append([close, revenue, date])
    if trade_info:
        trade_info.append(date)
        trade_info[0] = revenue - trade_info[0]
        trade_record.append(trade_info)
    return signal, trade_record

def boll(df, asset_type: str):
    '''Boll Strategy'''
    trades, num, buy_date, cash, revenue = 0, 0, 0, 1, 1
    ema_12, ema_v = None, None
    signal, trade_info, trade_record = list(), list(), list()
    for (close, date, volume, op), std in zip(df[["Close", "Dates", "BuyVolume", "Open"]].values.tolist(), df["Close"].rolling(78).std().values.tolist()):
        if ema_12 == None: ema_12 = close
        if ema_v == None: ema_v = volume
        ema_12 = ema_12 * 76 / 78 + close * 2 / 78
        ema_v = ema_v * 76 / 78 + volume * 2 / 78
        low_band, high_band = ema_12 - 2 * std, ema_12 + 2 * std
        if volume >= 1.5 * ema_v and op < low_band and close > low_band and cash:
            num += cash / close
            cash = 0 
            trade_info = [cash + num * close, date]
        elif op > high_band and close < high_band and num:
            cash += num * close
            num = 0
            trade_info.append(date)
            trade_info[0] = cash + num * close - trade_info[0]
            trade_record.append(trade_info)
            trade_info = list()
        revenue = cash + num * close
        signal.append([close, revenue, date])
        
    if trade_info:
        trade_info.append(date)
        trade_info[0] = revenue - trade_info[0]
        trade_record.append(trade_info)
        
    return signal, trade_record

def BackTest(dfs):
    ret = dict()
    ret["auto"] = simulate(dfs, auto)
    ret["macd"] = simulate(dfs, macd)
    ret["boll"] = simulate({"crypto": dfs["crypto"], "index": dict()}, boll)
    return ret

def save(ret):
    for strategy, strategy_result in ret.items():
        for asset_type, asset_result in strategy_result.items():
            for ticker, df in asset_result.items():
                path = "../result/{}_{}_{}.csv".format(strategy, asset_type, ticker)
                df["Revenue"].write.option("header", "true").format("csv").save(path)
                if df["Record"]:
                    path = "../result/{}_{}_{}_record.csv".format(strategy, asset_type, ticker)
                    df["Record"].write.option("header", "true").format("csv").save(path)

if __name__ == "__main__":
    dfs = load()
    ret = BackTest(dfs)
    save(ret)