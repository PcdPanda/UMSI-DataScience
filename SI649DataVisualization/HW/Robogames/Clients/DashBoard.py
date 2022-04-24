import streamlit as st
import time, json
import numpy as np
import altair as alt
import pandas as pd
import Robogame as rg

# define the options for widget
option_N = ["Arakyd Vocabulator Model", "Axial Piston Model", "Nanochip Model"]
option_Q = ["Astrogation Buffer Length", "InfoCore Size", "AutoTerrain Tread Count", "Polarity Sinks", "Cranial Uplink Bandwidth", "Repulsorlift Motor HP", "Sonoreceptors"]


# let's create two "spots" in the streamlit view for our charts

status = st.empty()
robotsTable = st.empty()
predVis = st.empty()
debug = st.empty()
# put input widget
hint_parts = st.multiselect("Select the part hint you want for hints", option_Q + option_N)
hint_robots = st.text_input("Input the robot id with interest, split by space", value="").split(" ")
robots = list()
for robot_id in hint_robots:
    if robot_id: robots.append(int(robot_id))
heatmap_Q = st.selectbox(label="Select the field for quantitative heatmap", options=option_Q)
partVis_Q = st.empty()
heatmap_N = st.selectbox(label="Select the field for nominal heatmap", options=option_N)

partVis_N = st.empty()


# create the game, and mark it as ready
game = rg.Robogame("bob")
game.setReady()


# wait for both players to be ready
while(True):    
    gametime = game.getGameTime()
    timetogo = gametime['gamestarttime_secs'] - gametime['servertime_secs']
    
    if ('Error' in gametime):
        status.write("Error"+str(gametime))
        break
    if (timetogo <= 0):
        status.write("Let's go!")
        break
    status.write("waiting to launch... game will start in " + str(int(timetogo)))
    time.sleep(1) # sleep 1 second at a time, wait for the game to start

info = dict() # The data will be stored in info dictionary

partInterest = hint_parts
robotInterest = robots

# Draw the scatter plot （expires - info_count）
def vis_priority():
    df_hints = info["partHints"].drop_duplicates().groupby(["id"]).count()["value"]
    df_hints.name = "info_count"
    robots_info = info["robots"].copy()
    robots_info["info_count"] = df_hints
    robots_info = robots_info.fillna(0)
    predhints_df = info["predHints"].sort_values(["time"]).drop_duplicates(subset=["id"], keep="last").set_index(["id"])
    robots_info["number"] = predhints_df["value"]
    robots_info["time"] = predhints_df["time"]
    chart = alt.Chart(robots_info.reset_index(), width=750, height=250).mark_circle().encode(
        x=alt.X("expires:Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("info_count:Q", scale=alt.Scale(domain=[-1, 11])),
        color=alt.Color("Productivity:Q", scale=alt.Scale(scheme="redyellowblue", reverse=True, domain=[-100, 100])),
        size=alt.value(50),
        tooltip=["id:N", "expires:Q", "info_count:Q", "Productivity:Q", "time:Q", "number:Q"]
    )
    return chart.interactive()

def heatmap(field):
    height, width = 300, 300
    df = info["partHints"].drop_duplicates().pivot(columns="column", index="id", values="value")
    df = df.merge(info["robots"].set_index(["id"]), left_index=True, right_index=True).dropna()
    if df.empty or field not in df.columns:
        return "We have not gather information about {}".format(field)

    df = df[[field, "Productivity"]]
    if field in option_N:
        x= alt.X("{}:N".format(field))
    else:
        x = alt.X("{}:Q".format(field), bin=True)
    heat = alt.Chart(df, width=width, height=height).mark_rect().encode(
        x=x,
        y=alt.Y("Productivity:Q", bin=True),
        color=alt.Color("count(*):Q")
    )
    text = alt.Chart(df).mark_text().encode(
        x=x,
        y=alt.Y("Productivity:Q", bin=True),
        text=alt.Text("count(*):Q")
    )
    right = alt.Chart(df, height=height, width=width / 5).mark_bar().encode(
        x=alt.X("count():Q", title=None),
        y=alt.Y("Productivity:Q", bin=True),
    )
    top = alt.Chart(df, width=width, height=height / 5).mark_bar().encode(
        x=x,
        y=alt.Y("count():Q", title=None),
    ) 
    return top & (((heat + text) | right))


for i in np.arange(0, 101):
    # this will give us just the new hints, but the object will store everything we've seen
    for t in np.arange(0,6):
        gametime = game.getGameTime()
        status.write("Seconds to next hack: {}\n Game remaining seconds: {} ".format(6-t, gametime["unitsleft"]))
        time.sleep(1)
    try:
        hints = game.getHints()
        info["robots"] = game.getRobotInfo()
        info["predHints"] = pd.DataFrame(game.getAllPredictionHints())
        info["partHints"] = pd.DataFrame(game.getAllPartHints())
        debug.write(pd.DataFrame(game.getAllPartHints()).shape)
    except Exception: pass
    # sleep 6 seconds
    
    info_robots = info["robots"]
    info_robots = info_robots[info_robots["expires"] >= gametime["curtime"]].sort_values(["expires"]).iloc[:10]

    game.setPartInterest(partInterest) # Set part interest
    game.setRobotInterest(robotInterest) # Set robot interest
    
    robotsTable.write(info_robots)
    predVis.write(vis_priority())
    partVis_Q.write(heatmap(heatmap_Q))
    partVis_N.write(heatmap(heatmap_N))

