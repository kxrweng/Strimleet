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

def generate_deviation(average_price, deviation_percentage=0.1):
    deviation = average_price * deviation_percentage * random.uniform(-1, 1)
    return average_price + deviation

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

# Ensure selected_cities is not None
selected_cities = state_city_data.get(selected_state, {"cities": []})

if "cities" in selected_cities:
    selected_city = st.sidebar.selectbox("City", selected_cities["cities"])
else:
    selected_city = st.sidebar.selectbox("City", [])

# Getting city numerical value from mapping
city_value = find_city_value(state_city_data, selected_city) if selected_city else None

# Getting state numerical value from mapping
state_value = find_state_value(state_city_data, selected_state)

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "random_points" not in st.session_state:
    st.session_state["random_points"] = None
if "point_predictions" not in st.session_state:
    st.session_state["point_predictions"] = None
if "average_prediction" not in st.session_state:
    st.session_state["average_prediction"] = None

if city_value is not None and state_value is not None:
    # Prepare input data for prediction
    input_data = np.array([[bedrooms, bathrooms, acrelot, house_size, state_value, city_value]])

    # Load the model and encoders from the pickle file
    with open('predict_model.pkl', 'rb') as file:
        data = pickle.load(file)


    rf_model = data["model"]

    def update_predictions():
        average_price = rf_model.predict(input_data)[0]
        st.session_state["average_prediction"] = average_price

        random_number = random.randint(1, 100)
        random_points = get_random_points(selected_state, selected_city, num_points=random_number)
        st.session_state["random_points"] = random_points

        if random_points is None:
            st.error("Could not generate random points. Please check the city and state values.")
            return

        # Generate predictions for each point
        st.session_state["point_predictions"] = [
            generate_deviation(average_price) for _ in random_points
        ]

    if st.sidebar.button("Predict"):
        update_predictions()

    if st.session_state["average_prediction"] is not None:
        st.write(f"Average Predicted Price: <span style='color:green; font-size:24px'>${st.session_state['average_prediction']}</span>", unsafe_allow_html=True)

    if st.session_state["random_points"]:
        random_points = st.session_state["random_points"]
        point_predictions = st.session_state["point_predictions"]

        # Initialize the map centered around the first point
        m = folium.Map(location=[random_points[0][0], random_points[0][1]], zoom_start=12)

        # Add points to the map
        for i, (lat, lon) in enumerate(random_points):
            folium.Marker(
                [lat, lon],
                popup=f"Predicted Price: ${point_predictions[i]:,.2f}"
            ).add_to(m)

        # Display the map in Streamlit
        st_folium(m, width=700, height=500)
    else:
        st.write("Choose a location to predict")
else:
    st.write("Please select valid state and city values to get predictions.")
