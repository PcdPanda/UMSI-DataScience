import streamlit as st
import time, json
import numpy as np
import altair as alt
import pandas as pd
import Robogame as rg
from altair.expr import datum

# let's create two "spots" in the streamlit view for our charts
status = st.empty()
cur_time = st.empty()
table_01 = st.empty()
predVis = st.empty()
partVis = st.empty()
facet_viz = st.empty()
test = st.empty()

selection = st.empty()

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

# run 100 times
for i in np.arange(0,101):
    # sleep 6 seconds
	for t in np.arange(0,6):
		status.write("Seconds to next hack: " + str(6-t))
		time.sleep(1)

	try:
		gametime = game.getGameTime()
		current_time = gametime['curtime']
	except:
		current_time = 'Game over :('
	cur_time.write("Current Time (XTU): " + str(current_time))

	robots = game.getRobotInfo()
	curr_bots = robots[robots['expires'] > current_time].drop(['winner', 'Productivity'], axis=1)
	expired_bots = robots[robots['expires'] < current_time]
	curr_bots = curr_bots.sort_values('expires')[:9]

	game.getHints(hintstart=0)
	pred_df = pd.DataFrame(game.getAllPredictionHints())
	df_merged = curr_bots.merge(pred_df, on="id", how="left")
	df_merged = df_merged.groupby(['id', 'name'], as_index=False).max()
	table_01.write(df_merged.sort_values('expires'))
    # update the hints
	#game.getHints()
	df2 = pd.DataFrame(game.getAllPartHints())

	# we'll want only the quantitative parts for this
	# the nominal parts should go in another plot
	quantProps = ['Astrogation Buffer Length','InfoCore Size',
		'AutoTerrain Tread Count','Polarity Sinks',
		'Cranial Uplink Bandwidth','Repulsorlift Motor HP',
		'Sonoreceptors']

	# if it's not empty, let's get going
	if (len(df2) > 0):
		df2 = df2[df2['column'].isin(quantProps)]
		facet_df = expired_bots.merge(df2, on="id")
		column_df = facet_df[['Productivity', 'column']]
		c2 = alt.Chart(facet_df).mark_point().encode(
			alt.X('Productivity:Q'),
			alt.Y('value:Q',scale=alt.Scale(domain=(-100, 100))),
			color='column:N',
		).properties(width=150,height=150).facet('column:N', columns=4)
	facet_viz.write(c2)

	heatmap_Q = st.selectbox(label="Select the field for quantitative heatmap", options=quantProps)

# 	df2 = pd.DataFrame(game.getAllPartHints())

# 	# we'll want only the quantitative parts for this
# 	# the nominal parts should go in another plot
# 	quantProps = ['Astrogation Buffer Length','InfoCore Size',
# 		'AutoTerrain Tread Count','Polarity Sinks',
# 		'Cranial Uplink Bandwidth','Repulsorlift Motor HP',
# 		'Sonoreceptors']

# 	# if it's not empty, let's get going
# 	if (len(df2) > 0):
# 		df2 = df2[df2['column'].isin(quantProps)]
# 		c2 = alt.Chart(df2).mark_circle().encode(
# 			alt.X('column:N'),
# 			alt.Y('value:Q',scale=alt.Scale(domain=(-100, 100)))
# 		)
# 		partVis.write(c2)

# 	# create a dataframe for the time prediction hints
# 	df1 = pd.DataFrame(game.getAllPredictionHints())

# 	# if it's not empty, let's get going
# 	if (len(df1) > 0):
# 		# create a plot for the time predictions (ignore which robot it came from)
# 		c1 = alt.Chart(df1).mark_circle().encode(
# 			alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
# 			alt.Y('value:Q',scale=alt.Scale(domain=(0, 100)))
# 		)

# 		# write it to the screen
# 		predVis.write(c1)

