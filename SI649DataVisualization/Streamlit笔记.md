# Streamlit笔记

### 1. 初始化

1. 安装streamlit

   ```python
   pip install streamlit==0.67 # 安装streamlit
   ```

2. 写相关脚本

   ```python
   # xxx.py
   import streamlit as st
   ```

3. 运行脚本

   ```python
   streamlit run xxx.py
   ```

### 2. 写文字 (write可以省略)

- 写标题

  ```python
  st.title("This is a title")
  ```

- 写md

  ```python
  st.write("## This is a markdown title")
  st.write("I'm a link to [document](https://docs.streamlit.io/en/stable/api.html)")
  ```

- 写dataframe

  ```python
  st.write(df)
  ```

- 画图

  ```python
  st.write(alt.Chart(df).mark_points().encode())
  ```

### 3. 绘图

- 创建按钮

  ```python
  btn = st.button("display the hp_mpg vis")
  if btn:
      st.write(hp_mpg)
  else:
      st.write("click the button")
  ```

- 创建选项栏

  ```python
  y_axis_option = ["Acceleration", "Miles_per_Gallon", "Displacement"]
  y_axis_select = st.selectbox(label="Select what y axis to be", options=y_axis_option)
  ```

- 创建侧边栏

  ```python
  st.sidebar.title("Sidebar Title")
  st.sidebar.write("#Sider stuff")
  ```

  

