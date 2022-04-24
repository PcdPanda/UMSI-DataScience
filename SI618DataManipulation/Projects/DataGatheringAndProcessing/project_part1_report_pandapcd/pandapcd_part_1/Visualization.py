import altair as alt
import pandas as pd
import pyspark.pandas as ps
alt.data_transformers.disable_max_rows()

def load():
    data = dict()
    prefix = "../result/"
    ans = ps.read_csv("{prefix}/ans.csv".format(**locals()))
    data["ans"] = pd.DataFrame(pd.DataFrame(ans.values, columns=ans.columns)).set_index(["Dates"])
    res = ps.read_csv("{prefix}/res.csv".format(**locals()))
    data["res"] = pd.DataFrame(pd.DataFrame(res.values, columns=res.columns)).set_index(["index"])
    records = ps.read_csv("{prefix}/records.csv".format(**locals()))
    data["records"] = pd.DataFrame(pd.DataFrame(records.values, columns=records.columns)).set_index(["index"])
    corr = ps.read_csv("{prefix}/corr.csv".format(**locals()))
    data["corr"] = pd.DataFrame(pd.DataFrame(corr.values, columns=corr.columns).set_index("index").stack()).reset_index().rename(columns={"index":"predictor", "level_1":"predicted", 0:"value"})
    predict = ps.read_csv("{prefix}/predict.csv".format(**locals()))
    data["predict"] = pd.DataFrame(pd.DataFrame(predict.values, columns=predict.columns).set_index("index").stack()).reset_index().rename(columns={"index":"predictor", "level_1":"predicted", 0:"value"})
    return data

def vis_price(df):
    df = df[["BTCUSDT", "ETHUSDT", "GOLD", "NSDQ", "SP500", "TNX"]]
    df = df.stack().reset_index().rename(columns={"level_1":"Strategy", 0:"Value"})
    selection = alt.selection_multi(fields=["Strategy"])
    color = alt.condition(selection, alt.Color("Strategy:N"), alt.value("lightgray"))
    selector = alt.Chart(df).mark_rect().encode(y=alt.Y("Strategy:N", title="Asset"), color=color).add_selection(selection)
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Dates:Q"),
        y=alt.Y("Value:Q"),
        tooltip=["Dates:Q", "Value:Q"],
        color=alt.Color("Strategy:N", title="Asset")
    ).transform_filter(selection).interactive(bind_y=False)
    return selector | chart

def vis_simulation(data, strategy_type):
    if strategy_type == "auto":
        columns = ['auto_BTCUSDT', 'auto_ETHUSDT', 'auto_GOLD', 'auto_NSDQ', 'auto_SP500',
       'auto_TNX']
    elif strategy_type == "macd":
        columns = ['macd_BTCUSDT', 'macd_ETHUSDT', 'macd_GOLD', 'macd_NSDQ','macd_SP500', 'macd_TNX']
    elif strategy_type == "boll":
        columns = ['boll_BTCUSDT', 'boll_ETHUSDT', "BTCUSDT", "ETHUSDT"]
    df = data["ans"][columns]
    df = df.stack().reset_index().rename(columns={"level_1":"Strategy", 0:"Value"})
    selection = alt.selection_multi(fields=["Strategy"])
    color = alt.condition(selection, alt.Color("Strategy:N"), alt.value("lightgray"))
    selector = alt.Chart(df).mark_rect().encode(y=alt.Y("Strategy:N"), color=color).add_selection(selection)
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Dates:Q"),
        y=alt.Y("Value:Q"),
        tooltip=["Dates:Q", "Value:Q"],
        color="Strategy:N"
    ).transform_filter(selection).interactive(bind_y=False)
    indicators = data["res"].loc[columns]
    indicators["SharpRatio"], indicators["CarmarRatio"] = (indicators["ret"] - 1.62) / indicators["std"], (indicators["ret"] - 1) / indicators["max_drawdown"]
    return selector | chart, indicators

def vis_corr(data):
    c1 = alt.Chart(data["corr"], width=500, height=500).mark_rect().encode(
        x=alt.X("predicted:N"),
        y=alt.Y("predictor:N"),
        color=alt.Color("value:Q")
    )
    t1 = alt.Chart(data["corr"]).mark_text().encode(
        x=alt.X("predicted:N"),
        y=alt.Y("predictor:N"),
        text=alt.Text("value:Q")
    )  
    c2 = alt.Chart(data["predict"], width=500, height=500).mark_rect().encode(
        x=alt.X("predicted:N"),
        y=alt.Y("predictor:N"),
        color=alt.Color("value:Q")
    )
    t2 = alt.Chart(data["predict"]).mark_text().encode(
        x=alt.X("predicted:N"),
        y=alt.Y("predictor:N"),
        text=alt.Text("value:Q")
    )
    return c1 + t1, c2 + t2

if __name__ == "__main__":
    data = load()
    q1 = vis_price(data["ans"].copy())
    price_indicators = data["res"].loc[["SP500", "TNX", "NSDQ", "GOLD", "ETHUSDT", "BTCUSDT"]]
    q2, auto_indicators = vis_simulation(data, "auto")
    q3, macd_indicators = vis_simulation(data, "macd")
    q4, boll_indicators = vis_simulation(data, "boll")
    q5, q6 = vis_corr(data)