import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.title("Data Plot")
#np.random.randn(100) 產生常態分佈的亂數
#cumsum 將這些數值累加
data = pd.DataFrame({"x":np.arange(1,101),
                     "y":np.random.randn(100).cumsum()})
st.line_chart(data)
st.line_chart(data,x="x",y="y")