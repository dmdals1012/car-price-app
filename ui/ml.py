
import streamlit as st
import pandas as pd
import joblib
import numpy as np



def run_ml() :
    # 유저에게 예측에 필요한 데이터를 입력받는다.
    # 나이, 연봉, 신용카드 부채, 순자산을 입력받는다.
    # 인공지능으로 예측하여, 결과를 화면에 보여준다.
    
    
    st.subheader('자동차 금액 예측')
    st.text('사용자의 정보를 입력하세요')

    age = st.number_input('나이를 입력하세요', min_value=0, max_value=120)
    salary = st.number_input('연봉을 입력하세요', min_value=0)
    credit_card_debt = st.number_input('신용카드 부채를 입력하세요', min_value=0)
    net_worth = st.number_input('순자산을 입력하세요', min_value=0)


    if st.button('예측하기') :
        regressor = joblib.load('model/regressor.pkl')
        input_data = np.array([age, salary, credit_card_debt, net_worth]).reshape(1,4)


        prediction = regressor.predict(input_data)

        pred_data = prediction[0]

        if pred_data < 0 :
            st.error('예측이 불가능한 데이터입니다.')
        else:
            # 소수점은 버리고 정수부분만 가져온다.
            pred_data = round(pred_data)
            # 숫자에 3자리마다 콤마를 찍어준다.
            pred_data = format(pred_data, ',')
            st.success(f'예측 금액은 {pred_data}$ 입니다.')


        