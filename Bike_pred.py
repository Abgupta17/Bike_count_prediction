import pickle
import streamlit as st
import numpy as np
import pickle


import os
from datetime import datetime
import pandas as pd

import streamlit as st
import base64

# Path to the background image
background_image_path = "C:/ALL IMPORTANT ML PROJ/South_korea_bike_prediction_model/bic_img.jpg"

# Load the background image
with open(background_image_path, "rb") as img_file:
    background_image = img_file.read()

# Convert the image to base64
image_base64 = base64.b64encode(background_image).decode()

# Apply the background image as CSS styling
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-image: url('data:image/jpg;base64,{image_base64}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


model_file_name = "trained_model.sav"
# model_path = r"C:\Users\Ab Gupta\Downloads\South_Korea_Seoul_Bike_prediction\Models\xgboost_regressor_r2_0_928_v1.pkl"

model = pickle.load(open(model_file_name, "rb"))

sc_dump_path = "sc.pkl"

sc = pickle.load(open(sc_dump_path, "rb"))

def season_to_df(seasons):
    seasons_cols = ['Spring', 'Summer', 'Winter']
    seasons_data = np.zeros((1, len(seasons_cols)), dtype="int")

    df_seasons= pd.DataFrame(seasons_data, columns=seasons_cols)
    if seasons in seasons_cols:
        df_seasons[seasons]=1
    return df_seasons

def get_string_to_datetime(date):
    dt = datetime.strptime(date, "%d/%m/%Y")
    return {"day": dt.day, "month": dt.month, "year": dt.year, "week_day": dt.strftime("%A")}


def days_df(week_day):
    days_names = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    days_name_data = np.zeros((1, len(days_names)), dtype="int")

    df_days = pd.DataFrame(days_name_data, columns=days_names)

    if week_day in days_names:
        df_days[week_day] = 1
    return df_days

def Bike_prediction(input_data):
    #changing the input_data to numpy array
    # input_data_as_numpy_array = np.asarray(input_data)
    #
    # # reshape the array as we are predicting for one instance
    # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    date, hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, seasons, holiday, functioning_day = input_data

    holiday_dic = {"No Holiday": 0, "Holiday": 1}
    finctioning_day = {"No": 0, "Yes": 1}

    str_to_date = get_string_to_datetime(date)

    u_input_list = [hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall,
                    holiday_dic[holiday], finctioning_day[functioning_day],
                    str_to_date["day"], str_to_date["month"], str_to_date["year"]]

    features_name = ["Hour", 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                     'Solar Radiation (MJ/m2)',
                     'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day', 'Month', 'Year', ]

    df_u_input = pd.DataFrame([u_input_list], columns=features_name)


    df_seasons = season_to_df(seasons)

    df_days = days_df(str_to_date["week_day"])

    df_for_pred = pd.concat([df_u_input, df_seasons, df_days], axis=1)



    sc_data_for_pred = sc.transform(df_for_pred)



    return f"Rented Bike Demand on date: {date}, and Time: {hour} is : {round(model.predict(sc_data_for_pred).tolist()[0])}"


def main():
    st.title('Bike Prediction System')
    st.markdown("---")

    # Input form with custom formatting
    with st.form("bike_form"):
        st.header('Enter Data')

        # Create two columns layout
        col1, col2, col3, col4 = st.columns(4)

        # First column
        with col1:
            for_date = st.date_input('Date', value=None)
            date = for_date.strftime("%d/%m/%Y")
            hour = st.slider('Hour', min_value=0, max_value=23)
            temperature = st.number_input('Temperature (°C)', value=0.0)



        # Second column
        with col2:
            wind_speed = st.number_input('Wind Speed (m/s)', value=0.0)
            holiday = st.selectbox('Holiday or Non-Holiday', ['No Holiday', 'Holiday'])
            visibility = st.number_input('Visibility (10m)', value=0.0)



        with col3:
            rainfall = st.number_input('Rainfall (mm)', value=0.0)
            snowfall = st.number_input('Snowfall (cm)', value=0.0)
            seasons = st.selectbox('Seasons', ['Spring', 'Summer', 'Fall', 'Winter'])



        with col4:
            humidity = st.number_input('Humidity (%)', value=0.0)
            solar_radiation = st.number_input('Solar Radiation (MJ/m2)', value=0.0)
            functioning_day = st.selectbox('Is it a functioning day?', ['No', 'Yes'])


        submit_button = st.form_submit_button(label='Bike Scenario')

    # Prediction result with custom formatting
    if submit_button:
        result = Bike_prediction(
            [date, hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall,
             seasons, holiday, functioning_day])
        st.success(result)


# Execute main function if script is run directly
    #
if __name__ == '__main__':
    main()






