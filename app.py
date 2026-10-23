
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="CO2 Emission Predictor",
    page_icon="🌱",
    layout="wide"
)

DATA_URL = "https://open.canada.ca/data/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/d589f2bc-9a85-4f65-be2f-20f17debfcb1/download/my2025-fuel-consumption-ratings.csv"

FEATURES = [
    "Engine size (L)",
    "Cylinders",
    "Combined (L/100 km)"
]

TARGET = "CO2 emissions (g/km)"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="latin1")
    df.columns = df.columns.str.strip()
    df = df[FEATURES + [TARGET]].copy()
    return df.dropna()


@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }

    return model, df, y_test, predictions, metrics


st.markdown("""
<style>
.stApp {
    background-color: #f6faf7;
    color: #1f2937;
}
[data-testid="stAppViewContainer"] {
    background-color: #f6faf7;
}
h1, h2, h3 {
    color: #166534 !important;
}
p, label, li {
    color: #26352b;
}
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d7e8da;
    padding: 16px;
    border-radius: 12px;
}
[data-testid="stMetricValue"] {
    color: #166534;
}
[data-testid="stSidebar"] {
    background-color: #eaf4ec;
}
.stButton > button {
    background-color: #15803d;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 42px;
}
.stButton > button p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("🌱 CO₂ Predictor")
    st.write("Machine Learning Project")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "Predict Emissions", "Model Analytics", "Dataset"]
    )
    st.divider()
    st.caption("Multiple Linear Regression")


st.title("CO₂ Emission Prediction")
st.write("Estimate vehicle emissions using Machine Learning.")
st.caption("Course-End Project | Environmental Data Analytics")
st.divider()

try:
    with st.spinner("Preparing the dataset and regression model..."):
        model, df, y_test, predictions, metrics = train_model()
except Exception as error:
    st.error("Unable to load the dataset or prepare the model.")
    st.write("Check your internet connection and try again.")
    st.exception(error)
    st.stop()


if page == "Home":
    st.subheader("Project Overview")
    st.write(
        "This application estimates vehicle carbon dioxide emissions "
        "from engine size, number of cylinders, and combined fuel consumption."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Input Features", "3")
    c3.metric("R² Score", f"{metrics['R² Score']:.3f}")

    st.subheader("How It Works")
    st.write("1. Vehicle data is loaded and cleaned.")
    st.write("2. A Multiple Linear Regression model is trained.")
    st.write("3. Vehicle details are used to predict CO₂ emissions.")
    st.write("4. Model performance is evaluated using test data.")

    st.info("Select **Predict Emissions** from the sidebar to try a prediction.")


elif page == "Predict Emissions":
    st.subheader("Enter Vehicle Details")
    st.write("Provide the vehicle specifications below.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            engine_size = st.number_input(
                "Engine Size (L)", 0.1, 10.0, 2.0, 0.1
            )

        with c2:
            cylinders = st.selectbox(
                "Number of Cylinders",
                [3, 4, 5, 6, 8, 10, 12, 16],
                index=1
            )

        with c3:
            fuel_consumption = st.number_input(
                "Combined Fuel Consumption (L/100 km)",
                1.0, 40.0, 8.0, 0.1
            )

        submitted = st.form_submit_button(
            "Predict CO₂ Emissions",
            use_container_width=True
        )

    if submitted:
        vehicle = pd.DataFrame(
            [[engine_size, cylinders, fuel_consumption]],
            columns=FEATURES
        )

        result = model.predict(vehicle)[0]

        st.subheader("Prediction Result")
        st.metric("Estimated CO₂ Emissions", f"{result:.2f} g/km")

        st.write("**Entered vehicle details**")
        st.dataframe(vehicle, use_container_width=True, hide_index=True)

        st.caption(
            "This is an estimate from the trained model, not a measured "
            "emission value. Accuracy may vary for different vehicles."
        )


elif page == "Model Analytics":
    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['R² Score']:.4f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f} g/km")
    c3.metric("RMSE", f"{metrics['RMSE']:.2f} g/km")

    st.write(
        "R² measures how much variation is explained by the model. "
        "MAE and RMSE measure prediction error; lower values are generally better."
    )

    st.subheader("Actual vs Predicted Emissions")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, predictions, alpha=0.65)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], "--")
    ax.set_xlabel("Actual CO₂ Emissions (g/km)")
    ax.set_ylabel("Predicted CO₂ Emissions (g/km)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


elif page == "Dataset":
    st.subheader("Dataset Preview")
    st.write("The table below shows the data used by the regression model.")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    st.write(f"Total usable records: **{len(df):,}**")
    st.write(f"Selected columns: **{', '.join(df.columns)}**")


st.divider()
st.caption("CO₂ Emission Prediction | Built with Python, Streamlit and Scikit-learn")
