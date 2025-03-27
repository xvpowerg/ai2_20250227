import pandas as pd
from datetime import datetime
import streamlit as st
def listToDict(dataList):
    dataDict = {}
    for i in range(len(dataList)):
        dataDict[dataList[i]] = i
    return dataDict
option_mult = st.multiselect("請選你喜歡的水果",["蘋果","香蕉","橘子","西瓜"])
st.write("你選擇的水果是:", option_mult)
dataList = ["春季","夏季","秋季","冬季"]
optionDict = listToDict(dataList)
option_radio = st.radio("你喜歡的季節:", dataList)
st.write("你選擇的季節是:", optionDict[option_radio], option_radio)
checkbox = st.checkbox("是否同意條款:")
if checkbox:
    st.write("你同意條款")
else:
    st.write("你不同意條款")
option_radio = st.select_slider(
    "選一個範圍",
    options = [1,2,3,4,5,6,7,8,9,10],
    value = (3,6)
)
st.write("你選擇的範圍是:", option_radio)
date_input = st.date_input("選一個日期:", datetime.today().date())
st.write("你選了:", date_input)