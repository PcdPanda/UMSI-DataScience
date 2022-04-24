from altair.vegalite.v4.schema.channels import Tooltip
from requests.api import options
import streamlit as st
import time, json
import numpy as np
import altair as alt
import pandas as pd
import Robogame as rg
from altair.expr import datum

# create the game, and mark it as ready
game = rg.Robogame("Untitled")
game.setReady()

info = dict()
partInterest = list()
robotInterest = list()

def gather_info():
	gameTime = game.getGameTime()
	if "Error" in gameTime:
		info["time"] = {"curtime":100, "unitsleft":0}
		return
	try:
		info["time"] = pd.DataFrame(pd.Series(gameTime)[["curtime", "unitsleft"]]).T
		hints = game.getHints(hintstart=0)
		partHints = pd.DataFrame(game.getAllPartHints()).drop_duplicates().pivot(
			columns="column", index="id", values="value"
		)
		predHints = pd.DataFrame(game.getAllPredictionHints()).set_index(["id"])
		all_robots = game.getRobotInfo().merge(partHints, left_index=True, right_index=True)
		all_robots["Part Hint Count"] = all_robots.count(axis=1) - 6
		info["All Robots"] = all_robots
		curtime = float(info["time"]["curtime"])
		robots = info["All Robots"].copy() # set the robot dataframe
		info["robots"] = robots[robots["expires"]>=curtime].sort_values(["expires", "Productivity"])
		info["num"] = info["robots"].merge(
			predHints, left_index=True, right_index=True
		).sort_values(["expires", "Productivity"])
	except Exception:
		pass

def trait_corr():
    df = info["All Robots"]
    df_corr = pd.DataFrame(columns=["Corr", "Num"])
    for column in quantProps:
        data = df[["Productivity", column]].dropna()
        corr = data["Productivity"].corr(data[column].astype(float))
        df_corr.loc[column] = [corr, len(data)]
    df_corr.fillna(0)
    df_corr.index.name="Traits"
    chart = alt.Chart(df_corr.reset_index()).mark_bar().encode(
        x=alt.X("Corr"),
        y=alt.Y("Traits:N", sort=alt.EncodingSortField(field='Corr', order='descending')),
        color=alt.Color("Num:Q"),
        tooltip=["Corr:Q","Num:Q"]
    ).properties(width=300, height=500).configure_axis(
    labelFontSize=10,
    titleFontSize=10
)
    return chart

def vis_priority():
	df = info['All Robots'][['id', 'expires', 'Part Hint Count', 'Productivity']].copy()
	df = df.fillna(0)
	chart = alt.Chart(df, width=750, height=250).mark_circle().encode(
        x=alt.X("expires:Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Part Hint Count:Q", scale=alt.Scale(domain=[-1, 11])),
        color=alt.Color("Productivity:Q", scale=alt.Scale(scheme="redyellowblue", reverse=True, domain=[-100, 100])),
        size=alt.value(50),
        tooltip=["id:Q", "expires:Q", "Part Hint Count:Q", "Productivity:Q"]
    )
	time_df = pd.DataFrame({"time":[float(info["time"]["curtime"])]})
	timeline = alt.Chart(time_df).mark_rule().encode(
        x = alt.X("time:Q"),
        size= alt.value(1),
        tooltip=["time:Q"]
    )
	return (chart + timeline).properties(title="All Robots over Time").interactive()

def trait_bars():
	robots = curr_bots.merge(part_df, left_index=True, right_index=True)
	robots = robots[robots['id'].isin(list(df_to_show['id'][:4]))]
	robots = robots.reindex(columns=['name','Astrogation Buffer Length', 'AutoTerrain Tread Count', 'Cranial Uplink Bandwidth', 'InfoCore Size', 'Polarity Sinks', 'Repulsorlift Motor HP', 'Sonoreceptors'])
	robots = pd.melt(robots, id_vars=['name'], value_vars=['Astrogation Buffer Length', 'AutoTerrain Tread Count', 'Cranial Uplink Bandwidth', 'InfoCore Size', 'Polarity Sinks', 'Repulsorlift Motor HP', 'Sonoreceptors'])
	chart = alt.Chart(robots).mark_bar().encode(
        x = alt.X('variable:N', title = None),
        y = alt.Y('value:Q', title = None),
        color = 'variable:N'
        ).properties(
            width = 150,
            height = 100
        ).facet(
            facet = 'name:N',
            columns = 5
        )
	return chart.properties(title="Traits Faceted by Name")

def trait_bars2():
	robots = curr_bots.merge(part_df, left_index=True, right_index=True)
	robots = robots[robots['name'].isin(list(df_to_show['name'][:4]))]
	robots = robots.reindex(columns=['name','Astrogation Buffer Length', 'AutoTerrain Tread Count', 'Cranial Uplink Bandwidth', 'InfoCore Size', 'Polarity Sinks', 'Repulsorlift Motor HP', 'Sonoreceptors'])
	robots = pd.melt(robots, id_vars=['name'], value_vars=['Astrogation Buffer Length', 'AutoTerrain Tread Count', 'Cranial Uplink Bandwidth', 'InfoCore Size', 'Polarity Sinks', 'Repulsorlift Motor HP', 'Sonoreceptors'])
	chart = alt.Chart(robots).mark_bar().encode(
        x = alt.X('name:N', title = None),
        y = alt.Y('value:Q', title = None),
        color = 'variable:N'
        ).properties(
            width = 150,
            height = 100
        ).facet(
            facet = 'variable:N',
            columns = 5
        )
	return chart

# let's create two "spots" in the streamlit view for our charts
status = st.empty()
cur_time = st.empty()
table_01 = st.empty()
option = st.selectbox(label="Select View Mode", options = ['Facet by Name','Facet by Trait'])
traitsChart = st.empty()
traitsHeatmap = st.empty()
test2 = st.empty()
facet_viz = st.empty()
placeholder = st.empty()
line_viz = st.empty()
table_entry = st.empty()

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

	gather_info()
	try:
		gametime = game.getGameTime()
		current_time = gametime['curtime']
	except:
		current_time = 100
	cur_time.write("Current Time (XTU): " + str(current_time))

	robots = game.getRobotInfo()

	curr_bots = robots[robots['expires'] > current_time].drop(['winner', 'Productivity'], axis=1)
	prod_bots = robots[~robots["Productivity"].isna()].drop(["winner"], axis=1)

	curr_bots = curr_bots.sort_values('expires')[:9]

	game.getHints(hintstart=0)
	part_df = pd.DataFrame(game.getAllPartHints()).drop_duplicates().pivot(
                columns="column", index="id", values="value"
            )

	props = ['id', 'name', 'expires', 
	'time', 'value', 'Arakyd Vocabulator Model', 
	'Axial Piston Model', 'Nanochip Model']
	pred_df = pd.DataFrame(game.getAllPredictionHints())
	df_merged = curr_bots.merge(pred_df, on="id", how="left").merge(part_df, on="id", how="left")
	df_merged2 = df_merged.groupby(['id', 'name'], as_index=False).max()

	df_to_show = df_merged[(
		df_merged['time'] < df_merged['expires']) | (df_merged['time'] < df_merged['expires'] + 10)].sort_values('expires')
	try:
		df_to_show = df_to_show[props]
	except:
		pass
	if len(df_to_show) > 0:
		df_to_show = df_to_show.groupby(['id', 'name'], as_index=False).max().sort_values('expires')
		table_01.write(df_to_show)
	else:
		table_01.write("Waiting for hints")

	next_five_bots = list(df_merged2.sort_values('expires')[:4]['id'])
	game.setRobotInterest(next_five_bots)

	if len(part_df) > 0:
		if 'Facet by Name' in option:
			traitsChart.write(trait_bars())
		elif 'Facet by Trait' in option:
			traitsChart.write(trait_bars2())
	
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
		facet_df = prod_bots.merge(df2, on="id")
		column_df = facet_df[['Productivity', 'column']]
		productivity_facet = alt.Chart(facet_df).mark_point().encode(
			alt.X('Productivity:Q'),
			alt.Y('value:Q',scale=alt.Scale(domain=(-100, 100))),
			color=alt.Color('column:N', legend=None),
		).properties(width=150,height=100).facet('column:N', columns=3)

	if len(prod_bots) > 0:
		with placeholder.container():
			col1, col2 = st.columns([4,1])
			col1.write(productivity_facet.properties(title="Productivity vs. Trait Value"))
			col2.write(trait_corr())
	if len(prod_bots) == 0:
		with placeholder.container():
			placeholder.write("Waiting for robots to expire for faceted chart")
	
	def trait_heatmap():
		fields = {'Arakyd Vocabulator Model', 'Axial Piston Model', 'Nanochip Model'}
		fields = list(fields & set(part_df.columns))
		if not fields: 
			return "Waiting for nominal traits to be updated"
		df = part_df[fields].copy()
		df = df.merge(prod_bots["Productivity"], left_index=True, right_index=True).reset_index().rename(columns={"index":"id"})

		charts = dict()
		width = 150
		height = 150
		for field in fields:
			y= alt.Y("{}:N".format(field))
			heat = alt.Chart(df, width=width, height=height).mark_rect().encode(
				x=alt.X("Productivity:Q", bin=True),
				y=y,
				color=alt.Color("count(*):Q", legend=None)
			)
			dots = alt.Chart(df, width=width, height=height).mark_circle(color="red").encode(
				x=alt.X("Productivity:Q"),
				y=y,
				tooltip=["Productivity:Q", "id:Q"]
			)
			text = alt.Chart(df).mark_text(dy=-10).encode(
				x=alt.X("Productivity:Q", bin=True),
				y=y,
				size=alt.value(15),
				text=alt.Text("count(*):Q")
			)
			charts[field] = heat + dots + text
		fields = list(set(fields) & set(charts.keys()))
		chart = charts[fields[0]]
		for i in range(1, len(fields)):
			chart = chart | charts[fields[i]]
		return chart.properties(title="Nominal Traits vs Productivity") 

	bots_merged = robots.merge(pred_df, on="id", how="left").drop_duplicates()
	curr_bot_ids = list(df_to_show['id'][:4])
	bots_merged = bots_merged[bots_merged['id'].isin(next_five_bots[:4])]

	c1 = alt.Chart(bots_merged).mark_line(point=True).encode(
		alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
		alt.Y('value:Q',scale=alt.Scale(domain=(0, 100))),
		tooltip=['id:Q', 'time:Q', 'value:Q']
	).properties(width=150, height=150)

	c3 = alt.Chart(bots_merged).mark_rule().encode(
		x='expires:Q'
	).properties(width=150, height=150)

	chart = alt.layer(c1, c3).facet('name:N', columns=4).properties(
		title="Number Hints by Robot")
	facet_viz.write(chart)
	line_viz.write(vis_priority())
	traitsHeatmap.write(trait_heatmap())

	# df = info['All Robots'].fillna(0)
	# test2.write(df)

	# test.write(vis_priority())

	# df1 = pd.DataFrame(game.getAllPredictionHints())

	# # if it's not empty, let's get going
	# if (len(df1) > 0):
	# 	# create a plot for the time predictions (ignore which robot it came from)
		# c1 = alt.Chart(df1).mark_circle().encode(
		# 	alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
		# 	alt.Y('value:Q',scale=alt.Scale(domain=(0, 100)))
		# )

	# 	# write it to the screen
	# 	test2.write(c1)

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

