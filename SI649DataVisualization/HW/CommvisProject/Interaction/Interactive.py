import pandas as pd
import altair as alt
import streamlit as st
from vega_datasets import data

st.beta_set_page_config(
    layout='centered', 
    initial_sidebar_state='collapsed', 
    page_title="I called this place 'America's worst place to live.' Then I went there."
)

df_amenity = pd.read_csv("amenity.csv")
df_economy = pd.read_csv("economy.csv")
df_bin = pd.read_csv("bin.csv")
df_life = pd.read_csv("life.csv", index_col=["State", "County"])


def v1(state: str="U.S.", field: str="Natural_Amenity"):
    df = df_amenity
    rank_field = "{}_State_Rank".format(field) if state != "U.S." else "{}_US_Rank".format(field)
    tooltip = ["County:N", "State:N", alt.Text("Natural_Amenity:Q", format=".3"), rank_field + ":Q"]
    lookup = [field, "County", "State", rank_field, "Natural_Amenity"]
    if field != "Natural_Amenity":
        y_field = "Standardized_{}".format(field)
        tooltip.insert(4, alt.Text("{}:Q".format(y_field), format=".3"))
        lookup.append(y_field)
    else: y_field = "Natural_Amenity"
    
    counties = alt.topo_feature(data.us_10m.url, 'counties')
    click = alt.selection_multi(fields=['id'])
    map_click = alt.selection_multi(fields=['id'])
    count = len(df) if state == "U.S." else len(df.set_index(["State"]).loc[state])
    title = "{} Worst and Best Counties' Performance in {}".format(state, field).replace("_", " ")
    bars = alt.Chart(df.reset_index(), title=title, width=600, height=200).transform_filter(
        "datum.{rank_field}<=20|datum.{rank_field}>={cutoff}".format(rank_field=rank_field, cutoff=count-20)
    )
    if state != "U.S.": bars = bars.transform_filter(
        alt.FieldOneOfPredicate(field="State", oneOf=[state]) 
    )
    bars = bars.mark_bar().encode(
        y=alt.Y("{}:Q".format(y_field)),
        opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
        color=alt.Color("{}:Q".format(y_field), scale=alt.Scale(scheme="redyellowblue", reverse=True)),
        x=alt.X('County', sort="y"),
        order=alt.Order("{}:Q".format(rank_field), sort='ascending'),
        tooltip=tooltip
    ).add_selection(click)
    title = "{} Geographic Distribution around U.S.".format(field).replace("_", " ")
    if field != "Natural_Amenity": title = "Standardized " + title
    county_map = alt.Chart(counties, width=600, height=350, title=title).mark_geoshape().transform_lookup(
        lookup='id',
        from_=alt.LookupData(df.reset_index(), 'id', lookup)
    ).encode(
        color="{}:Q".format(y_field),
        opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
        tooltip=tooltip
    ).project(type='albersUsa').add_selection(map_click)
    return bars & county_map 

def v2():
    height, width = 250, 350
    tooltip=["County:N", "State:N", "Gini:Q", "Median Child Income:Q", "Natural_Amenity:Q", "Unemployment Rate:Q"]
    bordertip = ["Gini_bin:O", "Employ_bin:O", "count():Q", "Avg Natural_Amenity:Q"]
    hover_bin = alt.selection_single(on="mouseover", 
        fields=["Gini_bin", "Employ_bin"], 
        init={"Gini_bin":0.2, "Employ_bin":5})
    hover_gini = alt.selection_single(on="mouseover", fields=["Gini_bin"], init={"Gini_bin":0.2})
    hover_employ = alt.selection_single(on="mouseover", fields=["Employ_bin"], init={"Employ_bin":5})
    na_min, na_max = df_economy["Natural_Amenity"].min(), df_economy["Natural_Amenity"].max()
    ub_slider = alt.binding_range(min=na_min, max=na_max, step=1, name="Natural Amenity Upper Bound")
    ub_selection = alt.selection_single(bind=ub_slider, fields=["ub"], init={"ub":na_max})
    lb_slider = alt.binding_range(min=na_min, max=na_max, step=1, name="Natural Amenity Lower Bound")
    lb_selection = alt.selection_single(bind=lb_slider, fields=["lb"], init={"lb":na_min})
    condition = alt.condition(# 使用过滤以保留小于slider的数据
        (alt.datum.Natural_Amenity <= ub_selection.ub) & (alt.datum.Natural_Amenity >= lb_selection.lb), 
        alt.value(1),
        alt.value(0.00)
    )
    income = alt.Chart(df_economy, width=width, height=height).mark_circle().encode(
        x=alt.X(
            "Gini:Q", 
            title="The lower Gini the better",
            scale=alt.Scale(domain=[0.15, 1.15], nice=False)
        ),
        y=alt.Y(
            "Unemployment Rate:Q", 
            title="The lower unemployment rate the better", 
            scale=alt.Scale(domain=[0, 25])),
        color=alt.Color(
            "Natural_Amenity:Q", 
            scale=alt.Scale(scheme="redyellowblue", reverse=True),
        ),
        opacity=condition,
        size=alt.value(25),
        tooltip=tooltip
    ).add_selection(hover_bin, ub_selection, lb_selection, hover_gini, hover_employ).interactive()
    heat = alt.Chart(df_bin, width=width, height=height).mark_bar(stroke='gray').encode(
        x=alt.X("Gini_bin:O", title="Groups with different Gini Index"),
        y=alt.Y("Employ_bin:O", 
            scale=alt.Scale(reverse=True), 
            title="Groups with different unemployment"),
        color=alt.Color(
            "Count:Q",
            scale = alt.Scale(scheme="goldgreen"),
            title="Number of counties in the group"
        ),
        tooltip=bordertip
    )
    text = alt.Chart(df_bin).mark_text().encode(
        x=alt.X("Gini_bin:O"),
        y=alt.Y("Employ_bin:O"),
        text=alt.Text("Count:Q")
    )
    border = alt.Chart(df_bin).mark_rect(color="red", filled=False).encode(
        x=alt.X("Gini_bin:O"),
        y=alt.Y("Employ_bin:O"),
    ).transform_filter(hover_bin) 
    unemployment = alt.Chart(df_bin, height=height, width=30).mark_bar().encode(
        x=alt.X("count():Q", title=None),
        y=alt.Y("Employ_bin:O", title=None, scale=alt.Scale(reverse=True)),
        opacity=alt.condition(hover_employ, alt.value(1), alt.value(0.25))
    )
    gini = alt.Chart(df_bin, width=width, height=30).mark_bar().encode(
        x=alt.X("Gini_bin:O", title=None),
        y=alt.Y("count():Q", title=None),
        opacity=alt.condition(hover_gini, alt.value(1), alt.value(0.25))
    ) 
    chart = (gini & 
             ((heat + text + border)|unemployment) & 
             income).resolve_scale(color='independent')
    chart = chart.properties(title={
        "text": ["Distribution of counties in different unemployment rate and Gini index interval"],
        "subtitle": ["Where is your home? Is it better than Red Lake?", "Find your county in the scatter plot through natural amenity, and see its group in the heatmap."]
    }).configure_title(anchor="start", fontSize=18)
    return chart
    

def v3(field="Annual_Crime_Rate"):
    step = {"Annual_Crime_Rate":5e-5, "Children_Poverty_Rate":5, "Chlamydia_Rate":1e-3, "Bachelor_or_High_Degree_Rate":2, "Teenage_Birth_Rate":0.02, "Life_Expectancy":0.5}
    q1, q9 = df_life[field].quantile(0.01), df_life[field].quantile(0.99)
    tooltip = [":Q".format(field), "{}_Rank:Q".format(field)]
    base = alt.Chart(df_life.reset_index(), width=650).transform_filter(
        "datum.{field}>={q1} & datum.{field}<={q9}".format(**locals())
    )
    chart = base.mark_bar().encode(
        x=alt.X("{}:Q".format(field), 
                bin=alt.BinParams(maxbins=15, nice=False, extent=[q1-step[field], q9+step[field]]),
                scale=alt.Scale(domain=[q1-step[field], q9+step[field]])),
        y=alt.Y("count():Q"),
        tooltip=["count():Q"],
        opacity=alt.value(0.5)
    )
    data = {"U.S. Mean".replace("_", " "): [df_life[field].mean()], 
            "U.S. Median".replace("_", " "): [df_life[field].median()],
            "Red Lake County": [df_life[field].loc["MN", "RED LAKE COUNTY"]],
           }
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index":"type", 0:"value"})
    rule=alt.Chart(df).mark_rule().encode(
        x=alt.X("value:Q"),
        color=alt.Color("type:N"),
        size=alt.value(5),
        tooltip=["type:N", alt.Text("value:Q", format=".2")]
    )
    rank = len(df_life) - int(df_life[field+"_Rank"].loc["MN", "RED LAKE COUNTY"])
    title = "U.S. {}".format(field.replace("_", " "))
    if field == "Life_Expectancy": rank = len(df_life) - rank + 1
    rank = "Red Lake County is better than {} counties".format(rank)
    return (chart + rule).resolve_scale(x="shared").properties(title={
            "text": [title],
            "subtitle": [rank]
        }).configure_title(anchor="start")


st.title("I called this place 'America's worst place to live.' Then I went there.")
st.write("Citizens of Red Lake County, Minn are outraged after reading the article [Every county in America, ranked by scenery and climate](https://www.washingtonpost.com/news/wonk/wp/2015/08/17/every-county-in-america-ranked-by-natural-beauty/) because home is judged as the 'absoluate America's worst place to live.' According to the [U.S. Department of Agriculture's natural amenities index](https://www.ers.usda.gov/data-products/natural-amenities-scale/), Red Lake County came in at the very bottom among all U.S. counties based on measures of scenery and climate. Six factors are determining the natural amenity index, and they're standardized for the computation. For example, with higher temperatures in January and lower temperatures in July, the county will get a higher natural amenity score. Although Red Lake County is not the worst in every aspect, even not in Minnesota, it still gets the lowest score when we consider all factors. Through the following visualization, we list the counties with best and worst performances in certain aspects of each state, can you find your home and compare it with Red Lake County?")

amenity_options = ["Natural Amenity", "Jan Temperature", "July Temperature", "July Humidity"]
state_option = ["U.S."] + sorted(set(df_amenity.reset_index()["State"]))
state = st.selectbox(label="Select the state you want investigate", options=state_option)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

field = st.radio("Select a field related to amenity.", amenity_options)
st.write(v1(state, field.replace(" ", "_")))

st.write("I guess your home does way better than Red Lake, right? However, it's unfair and narrow to judge Red Lake County as the worst place to live only based on the natural amenity index.  Living quality depends on many things, such as the economy, education, safety, etc. The so-called 'worst' arouses people's interest and attention to the county. Reporters, data analysts are paying more effort than ever to find out whether the Red Lake County is that bad as described in the article.")
st.write("When people are choosing where to settle down, employment and equality of wealth can be their concerns, because it depends if you can make a decent living. And Red Lake County does a more outstanding job there than I can imagine. The county beats a majority of U.S. counties in the unemployment rate and Gini index.")
st.write("In the following visualization, we put counties into groups based on their Gini index and unemployment rate. Most counties' Gini indexes are from 0.25-0.55, and their unemployment rates are 2.5 to 12.5. To our surprise, Red Lake is in the group with the best performance in Gini and second-best performance in the unemployment rate. I guess people who live there are never worried about making a living.")
st.write(v2())
st.write("In addition to the basic economic index, Red Lake county is also in the front rank in many aspects, including safety, poverty, and health. Through the following visualization, we can see that Red Lake is way better than the U.S. average level.")
well_being_option = ["Annual Crime Rate", "Children Poverty Rate", "Chlamydia Rate", "Life Expectancy"]
field = st.selectbox(label="Select the well being field you want investigate", options=well_being_option)
st.write(v3(field.replace(" ", "_")))
st.write("Being worst at the natural amenity index doesn't cause any negative effect on Red Lake County. For local people, it has become a slogan for advertisement and a joke for tea chat. For outsiders, it motivates them to dig up more interesting information about the county. Finally, it turns out that Red Lake County is a pretty good place to live.")
