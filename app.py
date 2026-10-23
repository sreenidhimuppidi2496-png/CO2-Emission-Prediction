
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="CO2 Emission Prediction",
    page_icon="🌱",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "data.csv"

FEATURES = [
    "Engine size (L)",
    "Cylinders",
    "Combined (L/100 km)"
]

TARGET = "CO2 emissions (g/km)"



st.markdown(
    """
    <style>
    /* Main background and default text */
    .stApp {
        background-color: #F5F9F6;
        color: #24352B;
    }

    /* Main content text */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp span,
    .stApp div[data-testid="stMarkdownContainer"] {
        color: #24352B;
    }

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #176B43 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E2F0E7;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #176B43 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #C8E0D0;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] label {
        color: #426451 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #176B43 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #218653;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
    }

    div.stButton > button p {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        background-color: #176B43;
        border: 1px solid #176B43;
    }

    /* Input boxes */
    .stApp input,
    .stApp textarea {
        color: #24352B !important;
        background-color: #FFFFFF !important;
    }

    /* Dataframes and tables */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
    }

    /* Links */
    .stApp a {
        color: #087443 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}. "
            "Place data.csv in the same folder as app.py."
        )

    df = pd.read_csv(DATA_FILE, encoding="latin1")
    df.columns = df.columns.str.strip()

    rename_columns = {
        "Engine Size(L)": "Engine size (L)",
        "Engine Size (L)": "Engine size (L)",
        "Engine size": "Engine size (L)",
        "Fuel Consumption Comb (L/100 km)": "Combined (L/100 km)",
        "Fuel Consumption Comb(L/100 km)": "Combined (L/100 km)",
        "Fuel Consumption (Comb (L/100 km))": "Combined (L/100 km)",
        "Fuel Consumption(Comb (L/100 km))": "Combined (L/100 km)",
        "Combined (L/100 km)": "Combined (L/100 km)",
        "CO2 Emissions(g/km)": "CO2 emissions (g/km)",
        "CO2 Emissions (g/km)": "CO2 emissions (g/km)",
        "CO2 emissions (g/km)": "CO2 emissions (g/km)"
    }

    df = df.rename(columns=rename_columns)

    missing_columns = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns not found: {missing_columns}. "
            f"Columns in your CSV: {df.columns.tolist()}"
        )

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])

    df = df[
        (df["Engine size (L)"] > 0)
        & (df["Cylinders"] > 0)
        & (df["Combined (L/100 km)"] > 0)
        & (df[TARGET] >= 0)
    ]

    if len(df) < 10:
        raise ValueError(
            "The dataset contains too few valid rows to train the model."
        )

    return df.reset_index(drop=True)


@st.cache_resource
def train_model():
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "R2 Score": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }

    return model, df, X_test, y_test, predictions, metrics


try:
    model, df, X_test, y_test, predictions, metrics = train_model()
except Exception as error:
    st.error("Unable to load the dataset or prepare the model.")
    st.warning(
        "Check that data.csv exists beside app.py and contains "
        "the required vehicle and CO2 columns."
    )
    st.exception(error)
    st.stop()


st.sidebar.title("🌱 CO2 Predictor")
st.sidebar.write("Vehicle Emission Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Predict Emissions",
        "Model Analytics",
        "Dataset"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Multiple Linear Regression")


if page == "Home":

    st.title("🌱 CO2 Emission Prediction")
    st.subheader("Vehicle Emission Analysis Using Machine Learning")

    st.write(
        "This application predicts vehicle carbon dioxide emissions "
        "using Multiple Linear Regression. It uses vehicle engine size, "
        "number of cylinders, and combined fuel consumption to estimate "
        "CO2 emissions in grams per kilometre."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Vehicles", f"{len(df):,}")

    with col2:
        st.metric(
            "Average CO2 Emissions",
            f"{df[TARGET].mean():.2f} g/km"
        )

    with col3:
        st.metric(
            "Model R² Score",
            f"{metrics['R2 Score']:.3f}"
        )

    st.subheader("About the Project")

    st.write(
        "The project applies supervised machine learning to estimate "
        "vehicle emissions from numerical vehicle characteristics. "
        "The dataset is divided into training and testing sets to "
        "evaluate the model on unseen records."
    )

    st.subheader("Input Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.markdown("**Engine Size**")
        st.write("Engine displacement measured in litres.")

    with feature_col2:
        st.markdown("**Number of Cylinders**")
        st.write("The number of cylinders in the vehicle engine.")

    with feature_col3:
        st.markdown("**Combined Fuel Consumption**")
        st.write("Fuel consumption measured in L/100 km.")

    st.info(
        "Open the Predict Emissions page to estimate CO2 emissions "
        "for a vehicle."
    )


elif page == "Predict Emissions":

    st.title("🚗 Predict Vehicle CO2 Emissions")

    st.write(
        "Enter the vehicle details below to estimate its CO2 emissions."
    )

    col1, col2 = st.columns(2)

    with col1:
        engine_size = st.number_input(
            "Engine Size (L)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1
        )

        cylinders_min = int(df["Cylinders"].min())
        cylinders_max = int(df["Cylinders"].max())

        cylinders = st.number_input(
            "Number of Cylinders",
            min_value=max(1, cylinders_min),
            max_value=max(1, cylinders_max),
            value=min(max(4, cylinders_min), cylinders_max),
            step=1
        )

    with col2:
        fuel_min = float(df["Combined (L/100 km)"].min())
        fuel_max = float(df["Combined (L/100 km)"].max())

        fuel_consumption = st.number_input(
            "Combined Fuel Consumption (L/100 km)",
            min_value=max(0.1, fuel_min),
            max_value=max(fuel_min + 0.1, fuel_max),
            value=float(
                np.clip(
                    8.0,
                    max(0.1, fuel_min),
                    max(fuel_min + 0.1, fuel_max)
                )
            ),
            step=0.1
        )

    if st.button("Predict CO2 Emissions", use_container_width=True):

        input_data = pd.DataFrame(
            [[engine_size, cylinders, fuel_consumption]],
            columns=FEATURES
        )

        predicted_emission = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Estimated CO2 Emissions",
            f"{max(0, predicted_emission):.2f} g/km"
        )

        if predicted_emission < 0:
            st.warning(
                "The model returned a negative estimate. "
                "This can happen with linear regression outside "
                "the patterns represented in the training data."
            )
        elif predicted_emission < 120:
            st.success("The estimated emission level is relatively low.")
        elif predicted_emission < 200:
            st.info("The estimated emission level is moderate.")
        else:
            st.warning("The estimated emission level is relatively high.")

        st.caption(
            "This is a machine learning estimate, not an official "
            "vehicle emission measurement."
        )


elif page == "Model Analytics":

    st.title("📊 Model Analytics")

    st.write(
        "The following metrics evaluate the Multiple Linear Regression "
        "model on the testing dataset."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "R² Score",
            f"{metrics['R2 Score']:.4f}"
        )

    with metric_col2:
        st.metric(
            "Mean Absolute Error",
            f"{metrics['MAE']:.2f} g/km"
        )

    with metric_col3:
        st.metric(
            "Root Mean Squared Error",
            f"{metrics['RMSE']:.2f} g/km"
        )

    st.divider()

    st.subheader("Actual vs Predicted CO2 Emissions")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    ax.set_xlabel("Actual CO2 Emissions (g/km)")
    ax.set_ylabel("Predicted CO2 Emissions (g/km)")
    ax.set_title("Actual vs Predicted Values")
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Coefficients")

    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    })

    st.dataframe(
        coefficient_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"Intercept: {model.intercept_:.4f}"
    )

    st.caption(
        "R² measures how much variation the model explains. "
        "MAE and RMSE measure prediction errors; lower values are "
        "generally better. Performance depends on the dataset."
    )


elif page == "Dataset":

    st.title("📁 Vehicle Dataset")

    st.write(
        "Explore the dataset used to train and evaluate the prediction model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Valid Records", f"{len(df):,}")

    with col2:
        st.metric("Dataset Columns", f"{df.shape[1]}")

    with col3:
        st.metric(
            "Average Fuel Consumption",
            f"{df['Combined (L/100 km)'].mean():.2f} L/100 km"
        )

    st.subheader("Data Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        df[FEATURES + [TARGET]].describe().round(2),
        use_container_width=True
    )

    st.subheader("CO2 Emission Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(
        df[TARGET],
        bins=25,
        edgecolor="black"
    )

    ax.set_xlabel("CO2 Emissions (g/km)")
    ax.set_ylabel("Number of Vehicles")
    ax.set_title("Distribution of Vehicle CO2 Emissions")
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_vehicle_emissions.csv",
        mime="text/csv"
    )
