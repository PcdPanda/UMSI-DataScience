# Your Name
# si649f20 altair transforms 2

# imports we will use
import altair as alt
import pandas as pd
import streamlit as st

# Title
st.title("Lab6 by Chongdan Pan")

# Import data
df1 = pd.read_csv(
    "https://raw.githubusercontent.com/eytanadar/si649public/master/lab6/hw/data/df1.csv"
)
df2 = pd.read_csv(
    "https://raw.githubusercontent.com/eytanadar/si649public/master/lab6/hw/data/df2_count.csv"
)
df3 = pd.read_csv(
    "https://raw.githubusercontent.com/eytanadar/si649public/master/lab6/hw/data/df3.csv"
)
df4 = pd.read_csv(
    "https://raw.githubusercontent.com/eytanadar/si649public/master/lab6/hw/data/df4.csv"
)

# change the 'datetime' column to be explicitly a datetime object
df2["datetime"] = pd.DatetimeIndex(pd.to_datetime(df2["datetime"])).tz_localize(
    tz="US/Central"
)
df3["datetime"] = pd.DatetimeIndex(pd.to_datetime(df3["datetime"])).tz_localize(
    tz="US/Central"
)
df4["datetime"] = pd.DatetimeIndex(pd.to_datetime(df4["datetime"])).tz_localize(
    tz="US/Central"
)
# Sidebar


###### Making of all the charts


########Vis 1
def v1():
    ##TODO: replicate vis 1
    # Interaction requirement 2, change opacity when hover over
    hover_selection = alt.selection_single(on="mouseover")
    brush_selection_emojis = alt.selection_interval(encodings=["y"])
    brush_selection_text = alt.selection_interval(encodings=["y"])

    opacityCondition_bar = alt.condition(hover_selection, alt.value(1), alt.value(0.6))
    # Interaction requirement 3 and 4, create brushing filter
    ##Static Component - Bars
    bar = (
        alt.Chart(df1)
        .mark_bar(color="orange", opacity=0.6, height=15)
        .encode(
            x=alt.X("PERCENT:Q", axis=None),
            y=alt.Y(
                "EMOJI:N",
                axis=None,
                sort=alt.EncodingSortField(field="PERCENT", order="descending"),
            ),
            tooltip=alt.Tooltip("EMOJI", title=None),
            opacity=opacityCondition_bar,
        )
        .transform_filter(brush_selection_emojis)
        .transform_filter(brush_selection_text)
    )
    ##Static Component - Emojis
    emojis = (
        alt.Chart(df1)
        .mark_text(align="left", width=20, opacity=0.6)
        .encode(
            text=alt.Text("EMOJI:N"),
            y=alt.Y(
                "EMOJI:N",
                axis=None,
                sort=alt.EncodingSortField(field="PERCENT", order="descending"),
            ),
            opacity=opacityCondition_bar,
        )
        .add_selection(brush_selection_emojis)
    )
    ##Static Component - Text
    text = (
        alt.Chart(df1)
        .mark_text(align="left", width=20, opacity=0.6)
        .encode(
            text=alt.Text("PERCENT_TEXT:N"),
            y=alt.Y(
                "EMOJI:N",
                axis=None,
                sort=alt.EncodingSortField(field="PERCENT", order="descending"),
            ),
            opacity=opacityCondition_bar,
        )
        .add_selection(brush_selection_text)
    )
    ##Put all together
    chart = (
        (emojis | text | bar)
        .add_selection(hover_selection)
        .configure_view(strokeWidth=0)
        .resolve_scale(y="shared")
    )
    chart


########Vis 2
def v2():
    # TODO: replicate vis2
    # Zooming and Panning
    line_selection = alt.selection_interval(bind="scales", encodings=["x"])
    # vertical line
    selection = alt.selection_single(
        on="mouseover", nearest=True, init={"x": 1503792330000.0}
    )
    opacity_condition = alt.condition(selection, alt.value(1), alt.value(0))
    vLine = (
        alt.Chart(df2)
        .mark_rule(color="lightgray", size=4, opacity=0)
        .encode(x=alt.X("datetime:T"), opacity=opacity_condition)
        .add_selection(selection)
    )
    # interaction dots
    dots = (
        alt.Chart(df2)
        .mark_circle(color="black", size=70)
        .encode(
            x=alt.X("datetime"),
            y=alt.Y("tweet_count:Q", title="Four-minute rolling average"),
            tooltip=["tweet_count:Q", "datetime:T", "team:N"],
            opacity=opacity_condition,
        )
        .interactive(bind_y=False)
    )
    # Static component line chart
    line = (
        alt.Chart(df2)
        .mark_line(size=2.5)
        .encode(
            y=alt.Y(
                "tweet_count:Q",
                scale=alt.Scale(domain=[0, 5.5]),
                title="Four-minute rolling average",
            ),
            x=alt.X("datetime:T", title=None),
            color=alt.Color("team:N"),
        )
        .add_selection(line_selection)
    )
    # Put all together
    line + vLine + dots


########Vis3

# Altair version
def v3():
    ##TODO: replicate vis3
    emojis = sorted(set(df3["emoji"]))
    widget = alt.binding_radio(options=emojis, name="Select Emoji", debounce=10)
    line_selection = alt.selection_single(
        fields=["emoji"], init={"emoji": "🔥"}, bind=widget
    )
    circle_selection = alt.selection_interval(
        init={"x": [0, 0], "y": [0, 0]}, empty="none"
    )
    line = (
        alt.Chart(df3)
        .mark_line()
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
            ),
        )
        .add_selection(line_selection)
        .transform_filter(line_selection)
    )
    circles = (
        alt.Chart(df3)
        .mark_circle(color="black")
        .add_selection(circle_selection)
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
            ),
            opacity=alt.condition(circle_selection, alt.value(0.7), alt.value(0)),
        )
        .add_selection(circle_selection)
        .transform_filter(line_selection)
    )
    line + circles
    options = ["🔥", "😴"]
    option = st.radio(label="Select Emoji", options=options)
    line = (
        alt.Chart(df3)
        .mark_line()
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
            ),
        )
        .transform_filter(alt.FieldOneOfPredicate(field="emoji", oneOf=[option]))
    )
    circles = (
        alt.Chart(df3)
        .mark_circle(color="black")
        .add_selection(circle_selection)
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
            ),
            opacity=alt.condition(circle_selection, alt.value(0.7), alt.value(0)),
        )
        .add_selection(circle_selection)
        .transform_filter(alt.FieldOneOfPredicate(field="emoji", oneOf=[option]))
    )
    line + circles


# Streamlit widget version


########Vis4 BONUS OPTIONAL
def v4():
    # Altair version
    # Streamlit widget version
    emoji_selection = alt.selection_single(encodings=["y"])
    emoji = (
        alt.Chart(df4, title="legend", height=100)
        .mark_text(size=25)
        .encode(
            text=alt.Text("emoji:N"),
            y=alt.Y("emoji:N", axis=None, sort=["🤣", "😭"]),
            opacity=alt.condition(emoji_selection, alt.value(1), alt.value(0.005)),
        )
        .add_selection(emoji_selection)
    )
    line = (
        alt.Chart(df4, title="Tears were shed-of joy and sorrow")
        .mark_line()
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
            ),
            color=alt.Color("emoji:N", legend=None),
            opacity=alt.condition(emoji_selection, alt.value(1), alt.value(0)),
        )
    )
    chart = (line | emoji).configure_view(strokeWidth=0).resolve_scale(y="shared")
    chart
    options = ["🤣", "😭"]
    option = st.radio(label="Select Emoji", options=options)
    color = "blue" if option == "😭" else "orange"
    line = (
        alt.Chart(df4, title="Tears were shed-of joy and sorrow")
        .mark_line(color=color)
        .encode(
            x=alt.X("datetime:T", title=None, axis=alt.Axis(tickCount=5)),
            y=alt.Y(
                "tweet_count:Q",
                title="Four-minute rolling average",
                axis=alt.Axis(tickCount=5),
                scale=alt.Scale(domain=[0, 1.4]),
            ),
        )
        .transform_filter(alt.FieldOneOfPredicate(field="emoji", oneOf=[option]))
    )
    chart = (line).configure_view(strokeWidth=0).resolve_scale(y="shared")
    chart


##### Display graphs


vis_option = ["v1", "v2", "v3", "v4"]
vis_select = st.sidebar.selectbox(
    label="Select a visualization to display", options=vis_option
)
if vis_select == "v1":
    v1()
elif vis_select == "v2":
    v2()
elif vis_select == "v3":
    v3()
elif vis_select == "v4":
    v4()
