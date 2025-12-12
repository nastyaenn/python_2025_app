import streamlit as st
import numpy as np
import joblib

#загружаем модель
model = joblib.load("model.pkl")

st.set_page_config(page_title="Income Prediction", page_icon="💰")

#cайдбар
st.sidebar.title("О приложении")
st.sidebar.write(
    """
    Это приложение предсказывает, превысит ли доход человека 50k в год
    Введите данные, и модель сделает прогноз для вас!
    """
)

#заголовок
st.title("Прогноз дохода > 50K 💸")
st.write("Заполните параметры ниже:")

#ввод данных
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Возраст", 18, 100, 30)
    education = st.number_input("Уровень образования (education-num)", 1, 20, 10)
    hours = st.number_input("Рабочие часы в неделю", 1, 100, 40)
with col2:
    fnlwgt = st.number_input("Весовой коэффициент (fnlwgt)", 0, 2000000, 150000)
    cap_gain = st.number_input("Доход с капитала (capital-gain)", 0, 100000, 0)
    cap_loss = st.number_input("Потери капитала (capital-loss)", 0, 5000, 0)

#прогноз
if st.button("Предсказать"):
    x = np.array([[age, fnlwgt, education, cap_gain, cap_loss, hours]])
    pred = model.predict(x)[0]
    st.subheader("Результат:")
    if pred == 1:
        st.success("💰 Ваш доход вероятно превышает 50K")
    else:
        st.error("😢 Ваш доход, скорее всего, не превышает 50K")