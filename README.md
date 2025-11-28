# Energy CO2 Dashboard

A Python Streamlit dashboard to visualize and analyze carbon intensity data collected from a public API.

## Website Link
https://energyco2.streamlit.app/

## Features

* Fetches carbon intensity data from the [UK Carbon Intensity API](https://api.carbonintensity.org.uk/)
* Cleans data using **IQR** and **Z-score** outlier removal methods
* Applies a **smoothing window** to reduce noise
* Visualizes:

  * Raw vs cleaned carbon intensity (line chart)
  * Normal distribution curve of cleaned data
  * Summary statistics
* Allows downloading the cleaned data as a CSV file
* Interactive **sidebar** to select:

  * Start and end dates
  * Outlier removal method (IQR or Z-score)
  * Smoothing window size

## Installation

1. Clone this repository:


2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate


3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run Energy_Streamlit.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`) in a web browser.

## Notes

* The dashboard focuses on visualizing carbon intensity data only.

## Requirements

* Python 3.8+
* pandas
* numpy
* matplotlib
* streamlit
* requests
* scipy

