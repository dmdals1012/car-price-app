
import pandas as pd
import streamlit as st
import seaborn as sb
import matplotlib.pyplot as plt




def run_eda() :

    st.subheader('탐색적 데이터 분석')

    st.info('데이터를 분석합니다.')

    df = pd.read_csv('data/Car_Purchasing_Data.csv')
    st.dataframe(df)

    if st.button('통계 데이터 보기') :
        st.dataframe(df.describe())

    if st.checkbox('상관관계 분석'):
        st.dataframe(df.corr(numeric_only=True))

    st.info('상관관계 분석')
    menu = ['차트로 보기','수치로 보기']
    choice = st.radio('선택하세요', menu)

    if choice == menu[0] :
        fig1 = plt.figure()
        sb.heatmap(df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, cmap='coolwarm', linewidths=0.5)
        st.pyplot(fig1)
    elif choice == menu[1] :
        st.dataframe(df.corr(numeric_only=True))

    st.info('최대 / 최소 데이터 확인하기')
    menu2 = ['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    selected_column = st.selectbox('컬럼 선택', menu2)
    st.text('최댓값 데이터')
    st.dataframe(df.loc[df[selected_column] == df[selected_column].max(), ])
    st.text('최소값 데이터')
    st.dataframe(df.loc[df[selected_column] == df[selected_column].min(), ])


