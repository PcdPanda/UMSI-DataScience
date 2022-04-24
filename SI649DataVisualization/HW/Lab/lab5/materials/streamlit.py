import streamlit as st
import altair as alt
import pandas as pd

st.title("My Title")

st.write("# Markdown Title")
st.write("I'm a link to [document](https://docs.streamlit.io/en/stable/api.html)")
car_url = "https://raw.githubusercontent.com/altair-viz/vega_datasets/master/vega_datasets/_data/cars.json"

cars = pd.read_json(car_url)
st.write(cars)

hp_mpg = (
    alt.Chart(cars)
    .mark_circle(size=80, opacity=0.5)
    .encode(x="Horsepower:Q", y="Miles_per_Gallon:Q", color="Origin")
)

btn = st.button("display the hp_mpg vis")
if btn:
    st.write(hp_mpg)
else:
    st.write("click the button")

y_axis_option = ["Acceleration", "Miles_per_Gallon", "Displacement"]
y_axis_select = st.selectbox(label="Select what y axis to be", options=y_axis_option)


hp_mpg = (
    alt.Chart(cars)
    .mark_circle(size=80, opacity=0.5)
    .encode(x="Horsepower:Q", y=y_axis_select, color="Origin")
)
st.write(hp_mpg)
st.sidebar.title("Sidebar Title")
st.sidebar.write("#Sider stuff")
