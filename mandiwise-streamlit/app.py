import json
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "mandi_price_forecasting_pipeline.joblib"
META_PATH = BASE / "mandi_price_forecasting_metadata.json"
MANDI_PATH = BASE / "agronimy mandi dataset.csv"
WEATHER_PATH = BASE / "up_district_weather_2024_2025.csv"

st.set_page_config(
    page_title="MandiWise | Agricultural Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL LIGHT THEME (WITH FIXED DROPDOWN CONTRAST)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
}

.stApp {
    background: #ffffff !important;
    color: #17251f !important;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border-bottom: 1px solid #e5ebe7;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif;
    color: #17251f !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown, .stCaption {
    color: #475569 !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #f0f8f2, #e8f5ed);
    border: 1px solid #d5e8db;
    border-radius: 18px;
    padding: 28px 30px;
    margin: 8px 0 22px;
}

.eyebrow {
    color: #218354 !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: 36px;
    margin: 8px 0;
    color: #14271d !important;
}

.hero p {
    color: #43594d !important;
    margin: 0;
    font-size: 15px;
}

/* Summary cards */
.kpi {
    background: #ffffff;
    border: 1px solid #e1e8e3;
    border-radius: 14px;
    padding: 18px;
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(20, 40, 28, 0.03);
}

.kpi-label {
    color: #64748b !important;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.kpi-value {
    color: #17251f !important;
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-sub {
    color: #218354 !important;
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f8faf9 !important;
    border-right: 1px solid #e2e8f0 !important;
}

section[data-testid="stSidebar"] label p {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* =========================================================
   FIXED SELECTBOX & DROPDOWN VISIBILITY
   ========================================================= */
/* Main closed selectbox container */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}

/* Text and selected value inside the selectbox */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {
    color: #0f172a !important;
    font-weight: 500 !important;
    -webkit-text-fill-color: #0f172a !important;
}

/* Dropdown arrow icon */
div[data-baseweb="select"] svg {
    fill: #334155 !important;
    color: #334155 !important;
}

/* Hover and Focus state on selectbox */
div[data-baseweb="select"] > div:hover {
    border-color: #218354 !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: #218354 !important;
    box-shadow: 0 0 0 2px rgba(33, 131, 84, 0.18) !important;
}

/* Opened Dropdown Menu List (Popover) */
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
div[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1) !important;
}

/* Individual Dropdown Options */
li[role="option"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* Hovered or Selected Option in Dropdown */
li[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
    background-color: #ecfdf5 !important;
    color: #14532d !important;
    font-weight: 600 !important;
}

/* Date & Number Inputs */
.stDateInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* Primary button */
.stButton button[kind="primary"] {
    background: #218354 !important;
    color: #ffffff !important;
    border: 1px solid #218354 !important;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
}

.stButton button[kind="primary"] * {
    color: #ffffff !important;
}

.stButton button[kind="primary"]:hover {
    background: #176b43 !important;
    border-color: #176b43 !important;
    color: #ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #64748b !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #218354 !important;
    font-weight: 700 !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #218354 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e1e8e3 !important;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetric"] label {
    color: #64748b !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #17251f !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e1e8e3 !important;
    border-radius: 10px !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    mandi = pd.read_csv(MANDI_PATH)
    weather = pd.read_csv(WEATHER_PATH)

    required_mandi = [
        "State",
        "District Name",
        "Market Name",
        "Commodity",
        "Price Date",
        "Modal Price (Rs./Quintal)",
    ]

    required_weather = [
        "District Name",
        "Price Date",
        "Rainfall_mm",
        "Max_Temp_C",
        "Min_Temp_C",
    ]

    missing_mandi = [
        c for c in required_mandi if c not in mandi.columns
    ]
    missing_weather = [
        c for c in required_weather if c not in weather.columns
    ]

    if missing_mandi:
        raise ValueError(
            f"Mandi dataset is missing columns: {missing_mandi}"
        )

    if missing_weather:
        raise ValueError(
            f"Weather dataset is missing columns: {missing_weather}"
        )

    mandi["Price Date"] = pd.to_datetime(
        mandi["Price Date"], errors="coerce"
    )
    mandi["Modal Price (Rs./Quintal)"] = pd.to_numeric(
        mandi["Modal Price (Rs./Quintal)"], errors="coerce"
    )

    mandi.dropna(
        subset=["Price Date", "Modal Price (Rs./Quintal)"],
        inplace=True,
    )

    weather["Price Date"] = pd.to_datetime(
        weather["Price Date"], errors="coerce"
    )

    for col in ["Rainfall_mm", "Max_Temp_C", "Min_Temp_C"]:
        weather[col] = pd.to_numeric(
            weather[col], errors="coerce"
        )

    weather.dropna(
        subset=["Price Date", "District Name"],
        inplace=True,
    )

    # The trained model is for Uttar Pradesh data.
    mandi = mandi[
        mandi["State"].astype(str).str.strip().str.lower()
        == "uttar pradesh"
    ].copy()

    weather = weather.groupby(
        ["District Name", "Price Date"], as_index=False
    )[["Rainfall_mm", "Max_Temp_C", "Min_Temp_C"]].mean()

    return mandi, weather


# =========================================================
# MODEL FEATURES
# =========================================================
def make_features(
    history,
    weather_row,
    state,
    district,
    market,
    commodity,
):
    history = history.sort_values("Price Date")
    prices = history["Modal Price (Rs./Quintal)"].astype(float)

    if len(prices) < 30:
        raise ValueError(
            "At least 30 earlier price observations are required."
        )

    row = {
        "Lag_1": prices.iloc[-1],
        "Lag_7": prices.iloc[-7],
        "Lag_14": prices.iloc[-14],
        "Lag_30": prices.iloc[-30],
        "Rolling_Mean_7": prices.tail(7).mean(),
        "Rolling_Mean_14": prices.tail(14).mean(),
        "Rolling_Mean_30": prices.tail(30).mean(),
        "Rainfall_mm": float(weather_row["Rainfall_mm"]),
        "Max_Temp_C": float(weather_row["Max_Temp_C"]),
        "Min_Temp_C": float(weather_row["Min_Temp_C"]),
        "State": state,
        "District Name": district,
        "Market Name": market,
        "Commodity": commodity,
    }

    return pd.DataFrame([row])


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AGRICULTURAL PRICE INTELLIGENCE</div>
    <h1>🌾 MandiWise</h1>
    <p>Explore historical mandi prices and estimate prices seven
    calendar days ahead using your trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INITIALIZE
# =========================================================
try:
    model = load_model()
    mandi, weather = load_data()

    with open(META_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

except Exception as error:
    st.error(f"Could not load model or data: {error}")
    st.stop()


# =========================================================
# SIDEBAR MARKET FILTERS
# =========================================================
st.sidebar.markdown("## 🌾 MandiWise")
st.sidebar.caption("Market selection")
st.sidebar.divider()

districts = sorted(
    mandi["District Name"].dropna().astype(str).unique()
)

if not districts:
    st.error("No Uttar Pradesh district records were found.")
    st.stop()

district = st.sidebar.selectbox("District", districts)

district_df = mandi[
    mandi["District Name"].astype(str) == district
]

markets = sorted(
    district_df["Market Name"].dropna().astype(str).unique()
)

if not markets:
    st.warning("No markets found for this district.")
    st.stop()

market = st.sidebar.selectbox("Market", markets)

market_df = district_df[
    district_df["Market Name"].astype(str) == market
]

commodities = sorted(
    market_df["Commodity"].dropna().astype(str).unique()
)

if not commodities:
    st.warning("No commodities found for this market.")
    st.stop()

commodity = st.sidebar.selectbox("Commodity", commodities)

selected = market_df[
    market_df["Commodity"].astype(str) == commodity
].copy()

selected = selected.sort_values("Price Date")

st.sidebar.divider()
st.sidebar.caption(
    f"Model: {metadata.get('selected_model', 'Saved model')}"
)
st.sidebar.caption("Forecast horizon: 7 calendar days")


# =========================================================
# SUMMARY METRICS
# =========================================================
if selected.empty:
    st.warning("No price records found for this selection.")
    st.stop()

latest = selected.iloc[-1]
latest_price = float(latest["Modal Price (Rs./Quintal)"])
latest_date = latest["Price Date"]

history_min = selected["Price Date"].min()
history_max = selected["Price Date"].max()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">'
        f'Latest recorded price</div><div class="kpi-value">'
        f'₹{latest_price:,.0f}</div><div class="kpi-sub">'
        f'Per quintal</div></div>',
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">'
        f'Market observations</div><div class="kpi-value">'
        f'{len(selected):,}</div><div class="kpi-sub">'
        f'Selected commodity</div></div>',
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">'
        f'Latest data date</div><div class="kpi-value">'
        f'{latest_date:%d %b %Y}</div><div class="kpi-sub">'
        f'Historical observation</div></div>',
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        '<div class="kpi"><div class="kpi-label">'
        'Forecast horizon</div><div class="kpi-value">'
        '7 days</div><div class="kpi-sub">'
        'Calendar days ahead</div></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# DASHBOARD TABS
# =========================================================
tab_overview, tab_forecast, tab_details = st.tabs(
    ["📊 Market overview", "🔮 Price forecast", "ℹ️ Model details"]
)


# =========================================================
# HISTORICAL MARKET OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### Historical price movement")

    st.caption(
        f"{commodity} · {market}, {district} · "
        f"{history_min:%d %b %Y} – {history_max:%d %b %Y}"
    )

    chart_data = selected[
        ["Price Date", "Modal Price (Rs./Quintal)"]
    ].groupby("Price Date", as_index=False).mean(
        numeric_only=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data["Price Date"],
            y=chart_data["Modal Price (Rs./Quintal)"],
            mode="lines",
            name="Modal price",
            line=dict(color="#218354", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(33,131,84,0.08)",
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "₹%{y:,.2f}/quintal<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=15, b=10),
        height=390,
        xaxis_title="Observation date",
        yaxis_title="Modal price (₹/quintal)",
        hovermode="x unified",
        font=dict(color="#334155"),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf1ee")

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Market summary")
        st.write("**State:** Uttar Pradesh")
        st.write(f"**District:** {district}")
        st.write(f"**Market:** {market}")
        st.write(f"**Commodity:** {commodity}")

    with right:
        st.markdown("#### Data coverage")
        st.write(f"**First observation:** {history_min:%d %b %Y}")
        st.write(f"**Last observation:** {history_max:%d %b %Y}")
        st.write(f"**Rows in selection:** {len(selected):,}")


# =========================================================
# SEVEN-DAY PRICE FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### Generate a seven-day-ahead estimate")

    st.write(
        "Select an input observation date. The model target is the "
        "modal price exactly seven calendar days after that date."
    )

    date_options = sorted(
        selected["Price Date"].dropna().unique()
    )

    chosen_date = st.selectbox(
        "Input observation date",
        options=date_options,
        index=len(date_options) - 1,
        format_func=lambda value: pd.Timestamp(value).strftime(
            "%d %B %Y"
        ),
    )

    chosen_date = pd.Timestamp(chosen_date)
    target_date = chosen_date + timedelta(days=7)

    st.info(f"**Forecast target date:** {target_date:%d %B %Y}")

    if st.button(
        "🔮 Forecast price",
        type="primary",
        use_container_width=True,
    ):
        try:
            # Only observations strictly before the input date.
            history = selected[
                selected["Price Date"] < chosen_date
            ].copy()

            weather_match = weather[
                (weather["District Name"].astype(str) == district)
                & (weather["Price Date"] == chosen_date)
            ]

            if len(history) < 30:
                st.error(
                    "Not enough earlier observations to calculate "
                    "the required 30-observation features."
                )

            elif weather_match.empty:
                st.error(
                    "No weather observation exists for this district "
                    "and date. Prediction was not generated."
                )

            else:
                weather_row = weather_match.iloc[0]

                weather_values = np.array([
                    weather_row["Rainfall_mm"],
                    weather_row["Max_Temp_C"],
                    weather_row["Min_Temp_C"],
                ], dtype=float)

                if not np.isfinite(weather_values).all():
                    st.error(
                        "Weather features contain missing or invalid values."
                    )
                    st.stop()

                X = make_features(
                    history,
                    weather_row,
                    "Uttar Pradesh",
                    district,
                    market,
                    commodity,
                )

                expected_features = metadata.get("feature_columns")

                if expected_features is None:
                    expected_features = metadata.get(
                        "feature_columns_order"
                    )

                if expected_features:
                    X = X[list(expected_features)]

                prediction = float(model.predict(X)[0])

                baseline_rows = selected[
                    selected["Price Date"] == chosen_date
                ]

                if baseline_rows.empty:
                    st.error(
                        "No actual price is available on the selected "
                        "date, so the price change cannot be calculated."
                    )
                else:
                    current_price = float(
                        baseline_rows[
                            "Modal Price (Rs./Quintal)"
                        ].mean()
                    )

                    change = prediction - current_price

                    percent_change = (
                        (change / current_price) * 100
                        if current_price != 0
                        else 0.0
                    )

                    st.divider()
                    st.markdown("### 🔮 7-Day Price Outlook")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Price on input date",
                        f"₹{current_price:,.2f}/quintal",
                    )

                    c2.metric(
                        "Predicted price",
                        f"₹{prediction:,.2f}/quintal",
                        delta=f"₹{change:+,.2f}/quintal",
                        delta_color="normal",
                    )

                    c3.metric(
                        "Expected price change",
                        f"{percent_change:+.2f}%",
                        delta_color="normal",
                    )

                    if change > 0:
                        st.success(
                            f"📈 **Expected to increase** by "
                            f"₹{change:,.2f}/quintal "
                            f"({percent_change:+.2f}%) over 7 days."
                        )
                    elif change < 0:
                        st.error(
                            f"📉 **Expected to decrease** by "
                            f"₹{abs(change):,.2f}/quintal "
                            f"({percent_change:+.2f}%) over 7 days."
                        )
                    else:
                        st.info("➡️ The predicted price is unchanged.")

                    st.caption(
                        f"Forecast date: {target_date:%d %B %Y}. "
                        "This is an estimate, not a guaranteed market price."
                    )

                    forecast_chart = go.Figure()

                    forecast_chart.add_trace(
                        go.Bar(
                            x=["Input-date actual", "7-day prediction"],
                            y=[current_price, prediction],
                            marker_color=["#94a3b8", "#218354"],
                            text=[
                                f"₹{current_price:,.2f}",
                                f"₹{prediction:,.2f}",
                            ],
                            textposition="outside",
                        )
                    )

                    forecast_chart.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=320,
                        yaxis_title="Price (₹/quintal)",
                        showlegend=False,
                        font=dict(color="#334155"),
                    )

                    st.plotly_chart(
                        forecast_chart,
                        use_container_width=True,
                    )

                    st.warning(
                        "Verify that lag and rolling-feature calculations "
                        "exactly match the original model-training notebook "
                        "before relying on these forecasts."
                    )

        except Exception as error:
            st.error(f"Forecast could not be generated: {error}")


# =========================================================
# MODEL DETAILS
# =========================================================
with tab_details:
    st.markdown("### Model information")

    st.write(
        f"**Selected model:** "
        f"{metadata.get('selected_model', 'N/A')}"
    )

    st.write(
        f"**Target column:** "
        f"{metadata.get('target_column', 'N/A')}"
    )

    st.write(
        f"**Forecast horizon:** "
        f"{metadata.get('forecast_horizon_days', 7)} calendar days"
    )

    features = metadata.get("feature_columns")

    if features is None:
        features = metadata.get("feature_columns_order", [])

    with st.expander("View model input features"):
        if features:
            for feature in features:
                st.write(f"• {feature}")
        else:
            st.write("Feature list is not available in metadata.")

    st.caption(
        "Historical evaluation metrics describe past performance "
        "and do not guarantee future prediction accuracy."
    )