# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import ee
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from streamlit_folium import st_folium
import shap
import matplotlib.pyplot as plt



# Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

</style>
"""
# Custom CSS Styling
st.markdown("""
<style>

.main{
background:#f6fff4;
}
# button styling
.stButton>button{
background:#2E8B57;
color:white;
border-radius:10px;
height:50px;
font-size:18px;
font-weight:bold;
width:100%;
}

# button hover effect
.stButton>button:hover{
background:#1f6f43;
}

div[data-testid="metric-container"]{
background:white;
padding:15px;
border-radius:12px;
border:1px solid #dddddd;
box-shadow:0 2px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# Hide Streamlit Default Elements
st.markdown(hide, unsafe_allow_html=True)
# load data
@st.cache_resource
def load_artifacts():

    artifacts = joblib.load(
        "models/crop_recommendation.joblib"
    )

    return (
        artifacts["model"],
        artifacts["soil_encoder"],
        artifacts["target_encoder"],
        artifacts["feature_columns"]
    )

# Load Model and Encoders
best_model, soil_encoder, target_encoder, feature_columns = load_artifacts()
xgb_model = best_model.named_steps["xgb"]

explainer = shap.TreeExplainer(xgb_model)


# Earth Engine Initialization
try:
    ee.Initialize()
except:
    st.warning(
        "Google Earth Engine not initialized."
    )
    
# Earth Engine Soil Function
def get_earth_engine_soil(latitude, longitude):

    point = ee.Geometry.Point(
        [longitude, latitude]
    )

    clay_dataset = ee.Image(
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
    )

    sample = clay_dataset.sample(
        region=point,
        scale=250
    ).first()

    soil = sample.getInfo()

    clay = soil["properties"]["b0"]

    if clay >= 40:
        soil_type = "Clay Loam"
    elif clay >= 20:
        soil_type = "Clay Loam"
    elif clay >= 10:
        soil_type = "Loamy"
    else:
        soil_type = "Sandy Loam"

    return soil_type, clay

# weather_api_key
def get_weather(latitude, longitude):

    api_key = "7d6dab32cf28dfce51279dd7edbcdfed"

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "weather_condition": data["weather"][0]["main"],
        "weather_description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "clouds": data["clouds"]["all"]
    }
    
# Weather Icon Function   
def weather_icon(weather):
    icons = {
        "Clear":"☀️",
        "Clouds":"☁️",
        "Rain":"🌧️",
        "Snow":"❄️",
        "Thunderstorm":"⛈️",
        "Mist":"🌫️",
        "Fog":"🌫️",
        "Drizzle":"🌦️"
        }
    return icons.get(weather,"🌤️")

# Get Coordinates Function
def get_coordinates(city):
    api_key = "7d6dab32cf28dfce51279dd7edbcdfed"

    url = (
        f"http://api.openweathermap.org/geo/1.0/direct?"
        f"q={city}&limit=1&appid={api_key}"
    )

    response = requests.get(url)

    data = response.json()

    if len(data) == 0:
        return None

    return {
        "latitude": data[0]["lat"],
        "longitude": data[0]["lon"],
        "country": data[0].get("country", ""),
        "state": data[0].get("state", "")
    }
    

# Season Determination Function
season_le = LabelEncoder()

season_le.fit([
    "Winter",
    "Spring",
    "Summer",
    "Autumn",
    "Rainy",
    "Dry"
])
    
def determine_season(temperature, rainfall):

    if temperature <= 5:
        return "Winter"

    elif temperature <= 15:
        if rainfall >= 500:
            return "Spring"
        else:
            return "Autumn"

    elif temperature <= 25:
        if rainfall >= 700:
            return "Rainy"
        else:
            return "Summer"

    else:
        if rainfall >= 800:
            return "Rainy"
        else:
            return "Dry"
        
# Prediction Function
def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    soil_type,
    season
):

    input_data = pd.DataFrame({
        "N":[N],
        "P":[P],
        "K":[K],
        "temperature":[temperature],
        "humidity":[humidity],
        "ph":[ph],
        "rainfall":[rainfall],
        "soil_type":[soil_type],
        "season":[season]
    })

    input_data["NPK_mean"] = (
        input_data[["N","P","K"]]
        .mean(axis=1)
    )

    input_data["THI"] = (
        input_data["temperature"]*0.1
        + input_data["humidity"]
    )

    input_data["ph_category"] = pd.cut(
        input_data["ph"],
        bins=[0,5.5,6.5,7.5,14],
        labels=[
            "Acidic",
            "Slightly Acidic",
            "Neutral",
            "Alkaline"
        ]
    )

    input_data["rainfall_level"] = pd.cut(
        input_data["rainfall"],
        bins=[-1,100,200,500],
        labels=[
            "Low",
            "Moderate",
            "High"
            ]
        )
    
    st.write("Rainfall:", rainfall)
    st.write(
        "Rainfall Level:",
        input_data["rainfall_level"].iloc[0]
    )
    
    
    try:
        input_data["soil_type"] = soil_encoder.transform(
            input_data["soil_type"]
        )
        
        
        
    except ValueError:
        input_data["soil_type"] = soil_encoder.transform(
            ["Clay Loam"]
        )

    ph_le = LabelEncoder()
    ph_le.fit([
        "Acidic",
        "Slightly Acidic",
        "Neutral",
        "Alkaline"
    ])

    input_data["ph_category"] = (
        ph_le.transform(
            input_data["ph_category"]
            .astype(str)
        )
    )

    rain_le = LabelEncoder()
    rain_le.fit([
        "Low",
        "Moderate",
        "High"
    ])

    input_data["rainfall_level"] = (
        rain_le.transform(
            input_data["rainfall_level"]
            .astype(str)
        )
    )

    input_data["season"] = [season]

    input_data["season"] = (
        season_le.transform(
            input_data["season"]
        )
    )

    input_data = input_data[
        feature_columns
    ]

    # Prediction 
    prediction = best_model.predict(input_data)[0]
    probabilities = best_model.predict_proba(input_data)[0]
    crop = target_encoder.inverse_transform([prediction])[0]
    # scaled_input = best_model.named_steps["scaler"].transform(input_data)
    scaled_input = input_data
    return (
        crop,
        prediction,
        probabilities,
        input_data,
        scaled_input
    )
    

# Streamlit Application Layout
st.markdown("""
# 🌾 Smart Crop Recommendation and Planting Advisory System

#### This system recommends the most suitable crop based on soil properties, weather conditions, and environmental factors.

The application uses an XGBoost machine learning model, integrated with Google Earth Engine and OpenWeatherMap, to retrieve location-specific environmental information and generate personalised crop recommendations. It also provides Explainable AI (SHAP) insights to help users understand the key factors influencing each recommendation, supporting more informed and transparent agricultural decision-making.
""")

st.sidebar.title("🌾 Crop Recommendation")

st.sidebar.success(
"""
AI Powered Agriculture

✔ XGBoost

✔ Google Earth Engine

✔ OpenWeatherMap

✔ Explainable AI
"""
)

tab1, tab2 = st.tabs([
    "Manual Input",
    "Location Based"
])

# Manual Input Tab
with tab1:

    st.subheader(
        "Manual Recommendation"
    )

    N = st.slider(
        "Nitrogen (N)",
        0,
        150,
        50,
        key="N"
    )

    P = st.slider(
        "Phosphorus (P)",
        0,
        150,
        40,
        key="P"
    )

    K = st.slider(
        "Potassium (K)",
        0,
        250,
        50,
        key="K"
    )

    temperature = st.slider(
        "🌡 Temperature (°C)",
        -10.0,
        50.0,
        25.0,
        key="temperature"
    )

    humidity = st.slider(
        "💧 Humidity (%)",
        0,
        100,
        60,
        key="humidity"
    )

    ph = st.slider(
        "🌱 Soil pH",
        min_value=3.5,
        max_value=10.0,
        value=6.5,
        step=0.1,
        help="Select the soil pH level",
        key="ph"
        )

    rainfall = st.slider(
        "🌧 Rainfall (mm)",
        0,
        500,
        200,
        key="rainfall"
    )
    
    soil_type = st.selectbox(
        "Soil Type",
    [
        "Alluvial",
        "Black Soil",
        "Clay Loam",
        "Loamy",
        "Sandy",
        "Sandy Loam",
        "Volcanic Loam",
        "Well-drained Loamy"
    ],
    key="soil_type"
)
    
    season = st.selectbox(
        "Season",
    [
        "Winter",
        "Spring",
        "Summer",
        "Autumn",
        "Rainy",
        "Dry"
        ],
        key="season"
    )
    
    if st.button(
        "Predict Crop"
    ):
        crop, prediction, probabilities, input_df, scaled_input = predict_crop(
            N,P,K,
            temperature,
            humidity,
            ph,
            rainfall,
            soil_type,
            season
        )

        st.markdown(f"""
                    <div style="
                    background:#2E8B57;
                    padding:40px;
                    border-radius:20px;
                    text-align:center;
                    color:white;
                    ">

                    <h2>🌾 Recommended Crop</h2>

                    <h3>{crop.upper()}</h3>

                    

                    </div>
                    """,
                    unsafe_allow_html=True)
        
        

        # top 5 crops
        classes = target_encoder.inverse_transform(
            np.arange(len(probabilities))
            
            )

        results = pd.DataFrame({
            "Crop": classes,
            "Probability": probabilities
            })

        results = results.sort_values(
            "Probability",
            ascending=False
            
            )
        top5 = results.head(5)
        st.bar_chart(
            top5.set_index("Crop")
            )
        shap_values = explainer.shap_values(scaled_input)
        class_index = prediction
        values = shap_values[0, :, class_index]

        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": np.abs(values)
            })

        importance = importance.sort_values(
            "Importance",
            ascending=False
            )

        st.subheader("📊 Feature Importance")

        st.bar_chart(
            importance.set_index("Feature")
            )

        top_features = importance.head(5)
        st.subheader("📝 Why this recommendation?")

        for feature in top_features["Feature"]:
            value = input_df.iloc[0][feature]

            st.write(
                f"• {feature} ({value}) strongly influenced the recommendation."
                )
            if feature == "temperature":
                st.write(
                    f"🌡 Temperature ({value:.1f}°C) is favourable for {crop}."
                    )
            elif feature == "humidity":
                st.write(
                    f"💧 Humidity ({value:.1f}%) supports healthy growth."
                    )
            elif feature == "rainfall":
                st.write(
                    f"🌧 Rainfall ({value:.1f} mm) contributed positively."
                    )
            elif feature == "ph":
                st.write(
                    f"🧪 Soil pH ({value:.1f}) is suitable for {crop}."
                    )
            elif feature == "soil_type":
                st.write(
                    "🌱 Soil type provides suitable growing conditions."
                    )
        
            # SHAP Summary Plot
            fig, ax = plt.subplots(figsize=(8,4))

            shap.plots.bar(
                shap.Explanation(
                    values=values,
                    feature_names=feature_columns
                ),
                show=False
            )

            st.pyplot(fig)
        
# Location-Based Tab
with tab2:

    st.subheader(
        "Location Based Recommendation"
    )

    # 
    city = st.text_input(
        "Enter City",
        "Delhi"
        )
    
    # retrieve location coordinates    
    if st.button("Get Recommendation"):
        location = get_coordinates(city)
        if location is None:
            st.error("City not found")
            st.stop()

        latitude = location["latitude"]
        longitude = location["longitude"]
        

        st.success(
            f"Location Found: {city}"
        )
        
        
        soil_type, clay = get_earth_engine_soil(
            latitude,
            longitude
        )

        weather = get_weather(
            latitude,
            longitude
        )

        # temperature, humidity, rainfall extraction
        temperature = weather["temperature"]
        humidity = weather["humidity"]
        rainfall = weather["rainfall"]

        st.subheader("🌱 Soil Information")

        
        col1,col2 = st.columns(2)

        col1.metric(
            "🌱 Soil Type",
            soil_type
        )

        col2.metric(
            "🪨 Clay Content",
            f"{clay}%"
        )

        st.subheader("🌦 Weather Information")

        
        col1,col2,col3 = st.columns(3)

        col1.metric(
            "🌡 Temperature",
            f"{weather['temperature']}°C"
        )

        col2.metric(
            "💧 Humidity",
            f"{weather['humidity']}%"
        )

        col3.metric(
            "🌧 Rainfall",
            f"{weather['rainfall']} mm"
        )

        icon = weather_icon(weather["weather_condition"])

        st.markdown(
            f"## {icon} {weather['weather_condition']}"
            )
        st.write(f"Description: {weather['weather_description']}")

        col1,col2,col3 = st.columns(3)

        col1.metric(
            "☁ Cloud Cover",
            f"{weather['clouds']}%"
        )

        col2.metric(
            "💨 Wind",
            f"{weather['wind_speed']} m/s"
        )

        col3.metric(
            "🌍 Pressure",
            f"{weather['pressure']} hPa"
        )
        
        season = determine_season(temperature, rainfall)

        # predict crop label
        crop, prediction, probabilities, input_df, scaled_input = predict_crop(
            N=st.session_state.N,
            P=st.session_state.P,
            K=st.session_state.K,
            temperature=temperature,
            humidity=humidity,
            ph=st.session_state.ph,
            rainfall=rainfall,
            soil_type=soil_type,
            season=season
            )
        
        # confidence
        confidence = np.max(probabilities) * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.1f}%"
        )

        st.subheader("🌾 Crop Recommendation")

        st.markdown(f"""
                    <div style="
                    background:#2E8B57;
                    padding:40px;
                    border-radius:20px;
                    text-align:center;
                    color:white;
                    ">

                    <h2>🌾 Recommended Crop</h2>

                    <h3>{crop.upper()}</h3>

                    

                    </div>
                    """,
                    unsafe_allow_html=True)  