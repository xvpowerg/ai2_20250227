import streamlit as st
import pandas as pd
from datetime import datetime
def colorIndex(colorData,findColor="紅色"):
    index = -1
    for i in range(len(colorData)):
        if colorData[i] == findColor:
            return i
    return index

st.title("請選元素")
data = ["紅色","藍色","綠色"]
option_selectbox = st.selectbox("你喜歡甚麼顏色:",["紅色","藍色","綠色"])
st.write("你選擇的顏色是:",option_selectbox,colorIndex(data,option_selectbox))