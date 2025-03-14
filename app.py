import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Function to fetch historical financial data for a given symbol and date range
@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        data.columns.name = None  # Remove column name to avoid confusion
    # Use only closing prices and drop any missing values
    data = data[['Close']]
    data.dropna(inplace=True)
    data["Close"] = data["Close"].squeeze()  # Ensure Close is a Series
    return data

# Step 1: Simulate predictions for Model A and Model B with noise
def simulate_predictions(data):
    np.random.seed(42)  # For reproducibility
    data["Prediction_A"] = data["Close"].apply(lambda x: x * (1 + np.random.normal(0, 0.02)))
    data["Prediction_B"] = data["Close"].apply(lambda x: x * (1 + np.random.normal(0, 0.01)))
    return data

# Step 2: Split traffic between two model groups (A and B)
def split_traffic(data):
    num_samples = len(data)
    data["Group"] = np.random.choice(["A", "B"], size=num_samples, p=[0.5, 0.5])
    data["Forecast"] = np.where(data["Group"] == "A", data["Prediction_A"], data["Prediction_B"])
    data["Forecast"] = data["Forecast"].squeeze()
    data["Close"] = data["Close"].squeeze()
    return data

# Step 3: Run the experiment by computing errors and other metrics
def run_experiment(data):
    data["Error"] = abs(data["Close"] - data["Forecast"])
    data["Percentage_Error"] = (data["Error"] / data["Close"]) * 100

    # Calculate directional accuracy based on trend differences
    data["Actual_Trend"] = np.sign(data["Close"].diff())
    data["Predicted_Trend_A"] = np.sign(data["Prediction_A"].diff())
    data["Predicted_Trend_B"] = np.sign(data["Prediction_B"].diff())

    accuracy_a = (data["Predicted_Trend_A"] == data["Actual_Trend"]).mean() * 100
    accuracy_b = (data["Predicted_Trend_B"] == data["Actual_Trend"]).mean() * 100

    # Simulate latency measurements for each model
    start_time_a = time.time()
    _ = data["Prediction_A"].mean()
    latency_a = time.time() - start_time_a

    start_time_b = time.time()
    _ = data["Prediction_B"].mean()
    latency_b = time.time() - start_time_b

    return data, accuracy_a, accuracy_b, latency_a, latency_b

# Step 4: Analyze the results using a t-test on percentage error
def analyze_results(data):
    errors_a = data[data["Group"] == "A"]["Percentage_Error"]
    errors_b = data[data["Group"] == "B"]["Percentage_Error"]

    t_stat, p_value = ttest_ind(errors_a, errors_b, equal_var=False)
    return errors_a, errors_b, t_stat, p_value

# Step 5: Prepare results for display
def format_results(data, errors_a, errors_b, accuracy_a, accuracy_b, latency_a, latency_b, t_stat, p_value):
    result = f"""
### Group Sizes:
{data.groupby("Group").size().to_string()}

*Model A Average Percentage Error:* {errors_a.mean():.4f}%

*Model B Average Percentage Error:* {errors_b.mean():.4f}%

*Directional Accuracy - Model A:* {accuracy_a:.2f}%

*Directional Accuracy - Model B:* {accuracy_b:.2f}%

*Model A Latency:* {latency_a:.6f} sec

*Model B Latency:* {latency_b:.6f} sec

*T-Test Results:* T-Statistic = {t_stat:.4f}, P-Value = {p_value:.4f}
    """

    if p_value < 0.05:
        result += "\n\nModel B shows a statistically significant improvement over Model A. Deploy Model B."
    else:
        result += "\n\nNo significant difference between Model A and Model B. Further improvements are needed."

    return result

def main():
    st.title("Financial Model Comparison Experiment")
    
    st.markdown("This app compares two simulated prediction models (A and B) using historical S&P 500 data.")
    
    # Sidebar settings for experiment parameters
    st.sidebar.header("Experiment Settings")
    symbol = st.sidebar.text_input("Ticker Symbol", "^GSPC")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    if st.sidebar.button("Run Experiment"):
        st.write("Fetching data...")
        data = get_data(symbol, start_date, end_date)
        st.write("Data fetched successfully!")
        
        with st.spinner("Simulating predictions..."):
            data = simulate_predictions(data)
            st.write("Predictions simulated.")
            
        with st.spinner("Splitting traffic..."):
            data = split_traffic(data)
            st.write("Traffic split completed.")
            
        with st.spinner("Running experiment..."):
            data, accuracy_a, accuracy_b, latency_a, latency_b = run_experiment(data)
            st.write("Experiment run completed.")
            
        with st.spinner("Analyzing results..."):
            errors_a, errors_b, t_stat, p_value = analyze_results(data)
            st.write("Analysis completed.")
        
        # Optionally show a preview of the processed data
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
            
        # Format and display the results
        results_markdown = format_results(data, errors_a, errors_b, accuracy_a, accuracy_b, latency_a, latency_b, t_stat, p_value)
        st.markdown(results_markdown)
        
        # Scatterplot showing both predictions and actual close prices
        st.subheader("Scatterplot: Predictions vs. Actual")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data.index, data["Prediction_A"], label="Prediction A", color="blue", alpha=0.6)
        ax.scatter(data.index, data["Prediction_B"], label="Prediction B", color="red", alpha=0.6)
        ax.plot(data.index, data["Close"], label="Actual Close", color="green", linewidth=2, alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Scatterplot of Predictions vs. Actual Closing Price")
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
  
