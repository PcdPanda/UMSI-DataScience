# SI649 Altair笔记

### 1. 安装和导入

```shell
pip install altair
```

```python
import altair as alt
```

### 2. 条形图,点状图, 和热点图

- 条形图

  ```python
  chart = alt.Chart(df, width=100).mark_bar(height=30, color="red").encode(
  	x = alt.X("mean(x):Q"),
      y = alt.Y("Y:Q", bin=alt.BinParams(maxbins=20), axis=None) # 聚类柱状图
  )
  ```

- 散点图

  ```python
  chart = alt.Chart(df, width=100).mark_points(filled=False).encode(
  	x = alt.X("median(x):Q", scale=alt.Scale(domain=[0, 100])), #设置坐标轴范围
      y = alt.Y("Y:N", sort=sort) #在Y轴上排序
  )
  ```

- 热点图

  ```python
  alt.Chart(df).mark_rect().encode(
      x=alt.X("MPG", bin=True),
      y=alt.Y("HP", bin=True),
      color=alt.Color("count()")
  )
  ```

- 折线图

  ```python
  line = alt.Chart(movies).mark_line(color="orange").encode(
      y=alt.Y("Major_Genre:N"),
      x=alt.X("median(Production_Budget):Q")
  )
  ```

### 3. 筛选

- 设置坐标轴范围

  ```python
  y = alt.Y("median(x):Q", scale=alt.Scale(domain=[0, 100])) #设置坐标轴范围
  ```

- 排序

  ```python
  y = alt.Y("Y:N", sort=sort) #在Y轴上根据指定顺序排序
  y=alt.Y("model:N", sort=alt.EncodingSortField(
          field="MPG",
          order="descending"
      )) # 根据数值排序
  ```

- 添加标题

  ```python
  chart.properties(
      title={
          "text": ["Median Budget For Films Since 1990"],
          "subtitle": ["2013 dollars"]
      }.configure_title(anchor="start")
  ```

- 添加文字

  ```python
  text = chart.mark_text(align="left", dx=7).encode(
  	text = alt.Text("mean(x):Q", formatType="number", format="$s")
  )
  chart + text
  ```

### 4. 多图拼接和叠加

- 图片拼接

  ```python
  top_chart & (left_chart | right chart).resolve_scale(y="shared")
  # 左图和右图共享y轴
  ```

- 图片叠加

  ```python
  chart1 + chart2 # chart2会放在chart1上方,一定要注意顺序 
  # 快速叠加
  base = chart1.encode(y=alt.Y("Y")) # 创建基底
  bar = chart1.mark_bar(color="blue").encode(x=alt.X("mean(x)") # 叠加bar
  line = chart1.mark_line(color="organge").encode(x=alt.X("median(x)") # 叠加线条
  ```

- 画类似的子图

  ```python
  # 一共画4个子图
  # 各有两张图y轴是us_gross, worldwide_gross
  # 各有两张图x轴是rotten_tomatoes_rating和imdb_rating
  repeated = alt.Chart(movies).mark_point().encode(
      x=alt.X(alt.repeat("column"), type="quantitative"),
      y=alt.Y(alt.repeat("row"), type="quantitative")
  ).properties(
      width=200,
      height=200
  ).repeat(
      row = ["US_Gross", "Worldwide_Gross"],
      column = ["Rotten_Tomatoes_Rating", "IMDB_Rating"]
  )
  ```

- filter数据中的不同字段，分别画子图

  ```python
  # major_genre中有action,drama等类型,分别画每种类型发布电影和月的关系
  alt.Chart(movies).mark_point().encode(
      x=alt.X("month(Release_Date):O"),
      y="count()",
      color="Major_Genre:N"
  ).properties(height=200, width=200).facet(
      facet="Major_Genre:N",
      columns=5
  )
  ```

### 5. 画交互图片

- 简单交互,可以放大缩小

  ```python
  chart.interactive()
  ```

- 添加tooltip (鼠标放上去有额外信息)

  ```python
  middle.encode(tooltip=["Title:N", "Release_Date:T"]).interactive() 
  #显示title和日期
  
  ```

### 6. 数据处理

- 画图时聚合

  ```python
  x=alt.X(field="Production_Budget", aggregate="mean", type="quantitative")
  ```

- 画图后聚合转换,<u>不join会删除没使用的字段</u>

  ```python
  alt.Chart(movies).transform_aggregate( # transform_joinaggregate原字段可以使用
  	groupby=["Major_Genre"],
      mean_product_budget="mean(Production_Budget)" # 创建了新字段,但是原字段删除了
  ).mark_bar().encode(
      y=alt.Y("Major_Genre:N"),
      x=alt.X("mean_product_budget:Q") # 使用新字段
  )
  ```

- bin聚合

  ```python
  y = alt.Y("Y:Q", bin=alt.BinParams(maxbins=20), axis=None) # 简单聚类柱状图
  # 使用transformt_bin
  alt.Chart(movies).mark_bar(width=35).transform_bin(
      "Binned_IMDB_Rating", "IMDB_Rating"
  ).encode(
      x=alt.X("Binned_IMDB_Rating:Q"),
      y=alt.Y("mean(Production_Budget):Q")
  )
  ```

- 添加字段

  ```python
  alt.Chart(movies).transform_calculate(
      Revenue="datum.Worldwide_Gross-datum.Production_Budget", # 创建Q字段
      expensive_or_cheap="datum.Production_Budget > 200000000 ? 'expensive' : 'cheap'"
  ).mark_point().encode(x=alt.X("Major_Genre:N"),y=alt.Y("Revenue:Q"))
  ```

- 过滤

  - 大小过滤

    ```python
    alt.Chart(movies).transform_filter(
        "datum.IMDB_Votes>=500 & datum.IMDB_Rating>5"
    ) # 使用js语言添加filter
    
    cjart.transform_filter( #设置Q类型数据范围
        alt.FieldRangePredicate(field="IMDB_Rating", range=[5, 7]) 
    )
    ```

  - nominal过滤

    ```python
    chart.transform_filter(# 设置N类型数据范围
        alt.FieldOneOfPredicate(field="test_result", oneOf=["Good", "Dubious"]) 
    )
    ```

- 添加随机选择数据集

  ```python
  chart.transform_sample(10) # 随机取10个数据画图
  ```

- window transformation,画出数据的rank

  ```python
  alt.Chart(movies, width=600).transform_window(
      sort=[alt.SortField("IMDB_Rating", order="descending")],
      imdb_rank="rank(*)" # 排序后添加字段
  ).transform_filter("datum.imdb_rank <= 10").mark_point().encode(
      x=alt.X("imdb_rank:Q"),
      y=alt.Y("Title:N", sort=alt.EncodingSortField(field="imdb_rank", order="ascending")) # 按照新字段排序
  )
  ```

- transform_fold画出堆叠bar,类似于panda pivot,将不同字段分开

  ```python
  alt.Chart(movies_wide).transform_fold(
      ["US_Gross", "Worldwide_Gross"],
      as_=["Gross", "dollars"]
  ).mark_bar().encode(
      x=alt.X("Title:N"),
      y=alt.Y("dollars:Q"),
      color=alt.Color("Gross:N")
  )

### 7. Selection

- single/multi/interval
  - selection_single: 一次只能选中一个data
    - on="mouseover": 当鼠标放置时就会当做选中,而不需要click
    - nearest=True: 自动选中鼠标最近的数据
    - clear="click":click后取消选中
    - encodings=["color"]: 每次会选中相同颜色的
  - selection_multi: 可以通过shift+enter一次选很多
  - selection_interval: 可以通过选中框选中一个范围
    - init={"x":[60, 50], "y":[15, 30]}: 初始化选中范围
    - encodings=["x"]:每次可以选中指定x区间的
  
- 创建流程

  1. 创建简单的interactive图像

  2. 定义selection和显示条件

     ```python
     selection = alt.selection_single(empty="none") # 初始什么都不选
     colorCondition = alt.condition(selection, "Origin", alt.value("gray")) 
     # Origin字段会改变颜色
     sizeCondition = alt.condition(selection, alt.SizeValue(200), alt.SizeValue(50)) # 选中后会改变大小
     ```

  3. encode selection

     ```python
     hp_mpg.add_selection(
         selection # 添加selection
     ).encode(
         color = colorCondition, # 添加condition
         size = sizeCondition
     )
     ```

### 8. 添加互动选项栏

1. 选择使用字段

   ```python
origins = sorted(data["Origins"])
   ```

2. 定义widget(选项栏类型)

   - Dropdown Box 

     ```python
     widget = alt.binding_select(options=origins, name="Select origin: ") 
     # 定义选项栏和使用字段

   - Radio Button

     ```python
     widget = alt.binding_select(options=origins, name="Select origin: ") 
     # 定义radio button和使用字段
     ```

   - Check Box

     ```python
     widget = alt.binding_checkbox(name="hidecolor")
     ```

   - Slider

     ```python
     hp_min, hp_max = cars["Horsepower"].min(), cars["Horsepower"].max()
     slider = alt.binding_range(min=hp_min, max=hp_max, step=1, name="cutoff")
     # 设置slider范围,并创建新字段叫做cutoff
     selection = alt.selection_single(bind=slider, fields=["cutoff"], init={"cutoff":hp_max}) # 根据slider字段的数值选中数据
     hp_mpg.add_selection(selection).transform_filter(
         alt.datum.Horsepower < selection.cutoff # 使用过滤以保留小于slider的数据
     )
     ```

     

3. 添加选中后的条件和效果

   ```python
   selectionOrigin = alt.selection_single(fields=["Origin"], init={"Origin": origins[0]}, bind=widget) # 选项栏作用于Origin字段
   colorCondition = alt.condition(selectionOrigin, "Origin:N", alt.value("lightgray")) # 选中后的变化
   c = hp_mpg.add_selection(selectionOrigin).encode(
       color = colorCondition
   )
   ```

