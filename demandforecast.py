import pandas as pd
import streamlit as st
import pickle
import io
import matplotlib.pyplot as plt

# Load the models
with open('modelarima.pickle', 'rb') as file:
    arima_model = pickle.load(file)
with open('modelprophet.pickle', 'rb') as file:
    prophet_model = pickle.load(file)

# Create a function to get averaged predictions for a date range


def get_averaged_predictions(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)

    # For Prophet
    future = pd.DataFrame({'ds': date_range})
    prophet_forecast = prophet_model.predict(future)
    prophet_predictions = prophet_forecast['yhat'].values

    # For ARIMA
    arima_predictions = arima_model.forecast(steps=len(date_range))[0]

    # Averaging
    averaged_predictions = (prophet_predictions + arima_predictions) / 2

    return date_range, averaged_predictions


# Streamlit app
st.title('Averaged Forecast Model Deployment')

# Date input from the user
start_date = st.date_input("Enter the start date for your forecast:")
end_date = st.date_input("Enter the end date for your forecast:")

if st.button("Predict"):
    dates, predictions = get_averaged_predictions(start_date, end_date)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Date': dates,
        'Predicted Value': predictions
    })

    # Display the results using Streamlit's line chart
    st.line_chart(plot_data.set_index('Date'))

    # Save the plot using Matplotlib in PNG format for download
    fig, ax = plt.subplots()
    ax.plot(plot_data['Date'], plot_data['Predicted Value'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Averaged Forecast")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    st.download_button(
        label="Download plot as PNG",
        data=buf,
        file_name="predicted_plot.png",
        mime="image/png"
    )
