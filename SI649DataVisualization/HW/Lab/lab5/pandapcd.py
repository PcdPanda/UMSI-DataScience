# pandapcd
# si649f20 Altair transforms 2

# imports we will use
import streamlit as st
import altair as alt
import pandas as pd

datasetURL="https://raw.githubusercontent.com/eytanadar/si649public/master/lab5/assets/hw/movie_after_1990.csv"
movies_test=pd.read_csv(datasetURL, encoding="latin-1")

#Title
st.title("Lab5 by pandapcd")

### Making of all charts

# Visualization 1
base = alt.Chart(movies_test).transform_filter(
    "datum.year>=1990 & datum.year <= 1997"
).transform_joinaggregate(
    mean_budget = "mean(budget):Q",
    max_budget = "max(budget):Q",
    groupby=["year"]
).transform_calculate(
    high_budget="datum.budget >= 2 * datum.mean_budget ? 'High' : 'Low'"
).encode(x=alt.X("year:O"))
line = base.mark_line().encode(y=alt.Y("mean_budget:Q"))
dot = line.transform_filter(
    alt.FieldOneOfPredicate(field="high_budget", oneOf=["High"])
).mark_point(filled=True).encode(y=alt.Y("budget:Q"))
text = dot.mark_text(align="center", dy=-10).transform_filter(
    "datum.budget == datum.max_budget"
).encode(
    text=alt.Text("title:N")
)
Vis1 = (dot + line + text).properties(height=300, width=750)

# Visualization 2

base = alt.Chart(movies_test).transform_filter(
    alt.FieldOneOfPredicate(field="test_result", oneOf=['Passes Bechdel Test', 'Fewer than two women', 'Women only talk about men', "Women don't talk to each other"])
)
heat = base.mark_rect().encode(
    x=alt.X("rating:Q", bin=True),
    y=alt.Y("test_result:N"),
    color=alt.Color("count():Q")
)  
text = base.mark_text().encode(
    x=alt.X("rating:Q", bin=True),
    y=alt.Y("test_result:N"),
    text=alt.Text("count():Q")
)  
high_light = base.transform_joinaggregate(
    max_rating="max(rating)",
    groupby=["test_result"]
).transform_filter(
    "datum.max_rating>=9"
).mark_rect(color="red", filled=False).encode(
    x=alt.X("rating:Q", bin=True),
    y=alt.Y("test_result:N")
)  
Vis2 = (heat + text + high_light).properties(height=150, width=700)

# Visualization 3
 
dom = alt.Chart(movies_test).transform_filter(
    alt.FieldOneOfPredicate("country_binary", oneOf=["U.S. and Canada"]),
).encode(
    y=alt.Y("test_result:N")
).mark_bar(color="blue").encode(
    x=alt.X("mean(dom_gross)"),
    color=alt.Color("mean(dom_gross)")
)
text_dom = alt.Chart(movies_test).transform_filter(
    alt.FieldOneOfPredicate("country_binary", oneOf=["U.S. and Canada"])
).transform_window(
    sort=[alt.SortField("dom_gross", order="ascending")],
    gross_rank="rank(dom_gross)",
    groupby=["test_result"]
).transform_filter("datum.gross_rank==10").mark_text(align="left", x=10).encode(
    y=alt.Y("test_result:N"),
    text=alt.Text("title:N"),
)
dom += text_dom

inter = alt.Chart(movies_test).transform_filter(
    alt.FieldOneOfPredicate("country_binary", oneOf=["International"]),
).encode(
    y=alt.Y("test_result:N")
).mark_bar(color="blue").encode(
    x=alt.X("mean(int_gross)"),
    color=alt.Color("mean(int_gross)", scale=alt.Scale(range=["white", "darkred"]))
)
text_int = alt.Chart(movies_test).transform_filter(
    alt.FieldOneOfPredicate("country_binary", oneOf=["International"])
).transform_window(
    sort=[alt.SortField("int_gross", order="ascending")],
    int_rank="rank(int_gross)",
    groupby=["test_result"]
).transform_filter("datum.int_rank==10").mark_text(align="left", x=10, color="white").encode(
    y=alt.Y("test_result:N"),
    text=alt.Text("title:N"),
)
inter += text_int
Vis3 = (dom & inter).resolve_scale(x="shared", color="independent")

# Visualization 4

Vis4 = alt.Chart(movies_test).transform_fold(
    ["budget", "int_gross"],
    as_=["finance", "dollars"]
).mark_bar().encode(
    x=alt.X("mean(dollars):Q"),
    y=alt.Y("finance:N"),
    color=alt.Y("finance:N")
).properties(height=100, width=500)

### Display charts



vis_option = ["Vis1", "Vis2", "Vis3", "Vis4"]
vis_select = st.sidebar.selectbox(label="Select a visualization to display", options=vis_option)
eval("st.write({})".format(vis_select))