import streamlit as st
import pandas as pd
st.title("CSV Reading")
upload_file = st.file_uploader("上傳CSV檔",type="csv")
if upload_file is not None:
    #st.write("檔名:",upload_file.name)
    df = pd.read_csv(upload_file)
    st.subheader("原始數據")
    st.dataframe(df)
    st.subheader("數據描述")
    st.dataframe(df.describe())
    st.subheader("數據視覺化")
    st.line_chart(df)

    columns = st.selectbox("選擇顯示的列",df.columns)
    st.dataframe(df[columns])

    selected_cloums = st.multiselect("選擇顯示的列",df.columns)
    if selected_cloums:
        st.dataframe(df[selected_cloums])
    else:
        st.write("請至少選擇一列")    