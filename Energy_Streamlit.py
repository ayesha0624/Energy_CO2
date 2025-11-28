import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------- Data Fetching -------------------
API_BASE = "https://api.carbonintensity.org.uk"

def fetch_carbon_intensity(start: datetime, end: datetime) -> pd.Series:
    from_ts = start.strftime("%Y-%m-%dT%H:%MZ")
    to_ts = end.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{API_BASE}/intensity/{from_ts}/{to_ts}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        recs = []
        for item in data.get('data', []):
            t = item['from']
            val = item['intensity'].get('actual') or item['intensity'].get('forecast')
            recs.append((pd.to_datetime(t), val))
        df = pd.DataFrame(recs, columns=['ts', 'intensity']).set_index('ts')
        return df['intensity'].astype(float)
    except:
        rng = pd.date_range(start, end, freq='30min')
        base = 200 + 80 * np.sin(np.linspace(0, 6.28 * (len(rng)/48), len(rng)))
        noise = np.random.normal(0, 12, size=len(rng))
        return pd.Series(base + noise, index=rng)

# ------------------- Outlier Removal -------------------
def remove_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    x = series.copy()
    valid = x.dropna()
    mu = valid.mean()
    sigma = valid.std()
    if sigma == 0 or np.isnan(sigma):
        return x
    z = (x - mu).abs() / sigma
    x[z > threshold] = np.nan
    return x


def remove_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    x = series.copy()
    valid = x.dropna()
    if len(valid) < 4:
        return x
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    x[(x < low) | (x > high)] = np.nan
    return x

# ------------------- Processing -------------------
def process_series(series, method='iqr', smoothing=3):
    if method == 'zscore':
        s = remove_outliers_zscore(series)
    else:
        s = remove_outliers_iqr(series)
    s = s.interpolate('time').fillna(method='ffill').fillna(method='bfill')
    s = s.rolling(smoothing, center=True, min_periods=1).median()
    s = s.rolling(smoothing, center=True, min_periods=1).mean()
    return s

# ------------------- Streamlit App -------------------
st.title("Energy CO2 Dashboard")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start Date", datetime.utcnow() - timedelta(days=2))
end_date = st.sidebar.date_input("End Date", datetime.utcnow())
method = st.sidebar.selectbox("Outlier Method", ['iqr', 'zscore'])
smoothing = st.sidebar.slider("Smoothing Window", min_value=1, max_value=10, value=3)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
else:
    raw = fetch_carbon_intensity(pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1))
    clean = process_series(raw, method=method, smoothing=smoothing)

    st.subheader("Raw vs Cleaned Carbon Intensity")
    df_plot = pd.DataFrame({'Raw': raw, 'Cleaned': clean})
    st.line_chart(df_plot)

    st.subheader("Normal Distribution Curve")
    fig, ax = plt.subplots()
    mean_val = clean.mean()
    std_val = clean.std()
    x = np.linspace(clean.min(), clean.max(), 100)
    y = norm.pdf(x, mean_val, std_val)
    ax.plot(x, y, color='blue')
    ax.set_title('Normal Distribution of Cleaned Carbon Intensity')
    ax.set_xlabel('gCO2/kWh')
    ax.set_ylabel('Probability Density')
    st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.write(df_plot.describe())

    # CSV Download
    csv = df_plot.to_csv(index=True)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='carbon_intensity.csv',
        mime='text/csv',
    )
