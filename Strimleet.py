import json
import pickle

import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder

def find_city_value(json_data, city_name):
    for state, state_data in json_data.items():
        if "cities" in state_data:
            if city_name in state_data["cities"]:
                return state_data["cities"][city_name]
        elif isinstance(state_data, dict):
            result = find_city_value(state_data, city_name)
            if result is not None:
                return result
    return None

def find_state_value(json_data, state_name):
    for state in json_data.keys():
        print(state)
        print(state_name)
        cleanedState = state.replace(" ", "")
        cleanedStateName = state_name.replace(" ", "")
        if(cleanedStateName == cleanedState):
            find_state_value(json_data, cleanedStateName)
    return "Not Found"

def find_state_value(json_data, state_name):
    for state, state_data in json_data.items():
        if state == state_name:
            return state_data.get("le_state", None)
    return None



# Streamlit app code
st.title("USA Real Estate Price Prediction")

st.subheader("USA Map")

# Load the JSON data
with open("state&city.json", "r") as json_file:
    state_city_data = json.load(json_file)

# Input form
st.sidebar.title("Input Features")
bedrooms = st.sidebar.number_input("Number of bedroom", value=3, min_value=2, max_value=5)
bathrooms = st.sidebar.number_input("Number of bathroom", value=2, min_value=1, max_value=4)
acrelot = st.sidebar.number_input("Acre lot (in acres)", value=1.0, min_value=0.0, max_value=1.2)
houseSize = st.sidebar.number_input("House Size (in square feet)", value=600, min_value=100, max_value=4363) 
# zipCode = st.sidebar.number_input("Zip Code", value=0, min_value=0, max_value=10) #Categorical Data
selected_state = st.sidebar.selectbox("State", list(state_city_data.keys()))
st.write(selected_state)
# Correct the line to populate selected_cities with city names
selected_cities = state_city_data.get(selected_state, [])

#Getting city numerical value from mapping [DONE]
selected_city = st.sidebar.selectbox("City", selected_cities["cities"])
city_value = find_city_value(state_city_data,selected_city)


#Getting state numerical value from mapping [DONE]
state_value = find_state_value(state_city_data,selected_state)
st.write(state_value)

# x = np.array([[3, 3, 0.7, 3, 10, 3852]])
y = np.array([[bedrooms,bathrooms,acrelot,houseSize,state_value,city_value]])
# st.write(state_city_data)
# Load the model and encoders from the pickle file

with open('predict_model.pkl', 'rb') as file:
    data = pickle.load(file)

rf_model = data["model"]
le_state_loaded = data["le_state"]
le_city_loaded = data["le_city"]





# Hardcoded input for testing

# Make predictions
predictions = rf_model.predict(y)

Predict = st.sidebar.button("Predict")
if Predict:
    st.write(f"Predicted Price: <span style='color:green; font-size:24px'>{predictions[0]}</span>", unsafe_allow_html=True)

# st.map()

# TODO get the mark down latitude and Longitude
# st.markdown("<h2 style='font-size: 24px;'>Map of Selected Area</h2>", unsafe_allow_html=True)

# geolocator = Nominatim(user_agent="my_geocoder")
# location = geolocator.geocode("selected_city, USA")
# print(location.latitude, location.longitude)

# st.map(location.latitude, location.longitude)


