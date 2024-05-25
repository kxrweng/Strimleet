import streamlit
import numpy as np
import sklearn
import folium
import streamlit_folium
import geopy
import shapely

print("streamlit version:", streamlit.__version__)
print("numpy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("folium version:", folium.__version__)
print("geopy version:", geopy.__version__)
print("shapely version:", shapely.__version__)

# Import and print the version of streamlit-folium
import pkg_resources

version = pkg_resources.get_distribution("streamlit_folium").version
print(version)
