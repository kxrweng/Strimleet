import json
import pickle
import numpy as np
import streamlit as st
import geopy.geocoders
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import random
from shapely.geometry import Point, box

def get_random_points(state, city, num_points=1):
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode(f"{city}, {state}")
    if location is None:
        return None
    
    lat, lon = location.latitude, location.longitude
    offset = 0.05
    bbox = box(lon - offset, lat - offset, lon + offset, lat + offset)
    
    minx, miny, maxx, maxy = bbox.bounds
    random_points = []
    for _ in range(num_points):
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        random_points.append((random_point.y, random_point.x))
    return random_points

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
        cleaned_state = state.replace(" ", "")
        cleaned_state_name = state_name.replace(" ", "")
        if cleaned_state_name == cleaned_state:
            return json_data[state].get("le_state", None)
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
house_size = st.sidebar.number_input("House Size (in square feet)", value=600, min_value=100, max_value=4363) 
selected_state = st.sidebar.selectbox("State", list(state_city_data.keys()))

# Correct the line to populate selected_cities with city names
selected_cities = state_city_data.get(selected_state, [])

# Getting city numerical value from mapping
selected_city = st.sidebar.selectbox("City", selected_cities["cities"])
city_value = find_city_value(state_city_data, selected_city)

# Getting state numerical value from mapping
state_value = find_state_value(state_city_data, selected_state)

# Prepare input data for prediction
input_data = np.array([[bedrooms, bathrooms, acrelot, house_size, state_value, city_value]])


# Load the model and encoders from the pickle file
with open('predict_model.pkl', 'rb') as file:
    data = pickle.load(file)

rf_model = data["model"]
# x = np.array([[3, 3, 0.7, 3, 10, 3852]])
# st.write("Trial Input : ", rf_model.predict(x) )


# Initialize session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "random_points" not in st.session_state:
    st.session_state["random_points"] = None

def update_predictions():
    st.session_state["predictions"] = rf_model.predict(input_data)
    st.session_state["random_points"] = get_random_points(selected_state, selected_city, num_points=100)

if st.sidebar.button("Predict"):
    update_predictions()

if st.session_state["predictions"] is not None:
    st.write(f"Predicted Price: <span style='color:green; font-size:24px'>{st.session_state['predictions'][0]}</span>", unsafe_allow_html=True)

if st.session_state["random_points"]:
    random_points = st.session_state["random_points"]
    # Initialize the map centered around the first point
    m = folium.Map(location=[random_points[0][0], random_points[0][1]], zoom_start=12)
    
    # Add points to the map
    for lat, lon in random_points:
        folium.Marker([lat, lon]).add_to(m)
    
    # Display the map in Streamlit
    st_folium(m, width=700, height=500)
else:
    st.write("Location not found")
