import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler
from visualizations import StockVisualizations
from data_manager import DataManager
from model_manager import ModelManager

@st.cache_resource
def initialize_components():
    """Initialize all components"""
    viz = StockVisualizations()
    data_manager = DataManager()
    model_manager = ModelManager()
    return viz, data_manager, model_manager


viz, data_manager, model_manager = initialize_components()

st.set_page_config(
    page_title="Stock Analysis & Investment Prediction",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("Stock Analysis & Investment Prediction")
    st.write("AI-driven stock predictions with real-time data analysis")
    
    st.sidebar.header("Controls")
    
    stock_options = {
        'Apple (AAPL)': 'AAPL',
        'Microsoft (MSFT)': 'MSFT',
        'Intel (INTC)': 'INTC',
        'IBM': 'IBM'
    }
    selected_stock = st.sidebar.selectbox("Choose a company:", list(stock_options.keys()))
    ticker = stock_options[selected_stock]
    
    tab1, tab2, tab3 = st.tabs(["Real-time Analysis", "Historical Analysis", "Model Insights"])
    
    with tab1:
        st.header("Real-time Analysis")
        
        if st.button("Fetch Latest Data", key="fetch_real_time"):
            with st.spinner("Fetching latest data..."):
                real_time_data = data_manager.fetch_real_time_data(ticker)
                
                if real_time_data is not None:
                    st.success(f"Latest data fetched for {ticker}")
                    
                    st.subheader("Latest Stock Data")
                    st.dataframe(real_time_data.tail())
                    
                    viz.plot_time_series(
                        real_time_data, 'Date', 'Close', 
                        f'{ticker} Close Price', f'{ticker} Latest Prices'
                    )

                    historical_data = data_manager.get_historical_data(ticker)
                    if historical_data is not None:
                        model, _, _, reverse_map = model_manager.get_or_train_model(ticker, historical_data)
                        if model is not None:
                    
                            features = data_manager.generate_features(real_time_data)
                            latest_features = features.tail(1)[[
                                'Close_Lag1', 'Close_Lag2', 'MA5', 'MA10',
                                'Momentum_5', 'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV'
                            ]]
                    
                            if not latest_features.isnull().values.any():
                                predicted_label = model_manager.predict(ticker, latest_features)
                        
                                if predicted_label is not None:
                                    action_map = {-1: "Sell", 0: "Hold", 1: "Buy"}
                                    recommendation = action_map[predicted_label]
                            
                                    st.subheader("Investment Recommendation")
                                    col1, col2 = st.columns(2)
                            
                                    with col1:
                                        st.metric(f"Recommendation for {ticker}", recommendation)
                            
                                    with col2:
                                        if hasattr(model, 'predict_proba'):
                                            scaler = StandardScaler()
                                            features_scaled = scaler.fit_transform(latest_features)
                                            proba = model.predict_proba(features_scaled)[0]
                                            confidence = max(proba)
                                            st.metric("Model Confidence", f"{confidence:.2%}")
                            
                                    guidance_map = {
                                        -1: f"Based on recent market behavior, it may be advisable to sell your position in {ticker}.",
                                        0: f"The model does not indicate a strong signal to buy or sell. Maintaining your current position in {ticker} may be the most prudent decision at this time.",
                                        1: f"Current indicators suggest that buying additional shares of {ticker} may be advantageous, given the model's prediction of upward movement."
                                    }
                            
                                    st.info(guidance_map[predicted_label])
                                else:
                                    st.warning("Unable to make prediction at this time.")
                            else:
                                st.warning("Not enough recent data for prediction.")
                        else:
                            st.error("Failed to train model. Please try again.")
                    else:
                        st.error("Historical data not available for training.")
                else:
                    st.error("Failed to fetch latest data. Please try again.")
    
    with tab2:
        st.header("Historical Analysis")
        
        historical_data = data_manager.get_historical_data(ticker)
        sp500_data = data_manager.get_historical_data('SP500')
        
        if historical_data is not None and sp500_data is not None:
            if st.checkbox("Show raw historical data"):
                st.dataframe(historical_data.tail())
            
            viz.plot_time_series(
                historical_data, 'Date', 'Close', 
                f'{ticker} Historical Close Price', f'{ticker} Historical Prices'
            )
            
            st.subheader("Comparison with S&P 500")
            viz.compare_to_sp500(historical_data, sp500_data, ticker)
            
            st.subheader("Rolling Correlation with Interest Rates")
            interest_data = data_manager.get_historical_data('InterestRates')
            if interest_data is not None:
                window = st.slider("Correlation Window (days)", 30, 180, 90)
                viz.plot_rolling_correlation(
                    historical_data, interest_data, window,
                    f'{ticker} Rolling Correlation with Interest Rates ({window} days)'
                )
    
    with tab3:
        st.header("Model Insights")
        
        st.subheader("Correlation Matrix")
        corr_matrix, merged_data = data_manager.calculate_correlation_matrix(ticker)
        
        if corr_matrix is not None:
            viz.plot_heatmap(corr_matrix, f'Correlation Matrix for {ticker} and Market Indicators')
            
            st.write("Correlation Values:")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
            
            st.subheader("Feature Importance")
            model_info = model_manager.models.get(ticker)
            if model_info is not None:
                model = model_info['model']
                if hasattr(model, 'estimators_'):
                    for name, estimator in model.estimators:
                        if hasattr(estimator, 'feature_importances_'):
                            feature_names = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10', 
                                          'Momentum_5', 'Momentum_10', 'Daily_Return', 
                                          'Volume_Lag1', 'OBV']
                            viz.plot_feature_importance(
                                feature_names, estimator.feature_importances_,
                                f'Feature Importance for {ticker}'
                            )
                            break
            
            st.subheader("Model Performance")
            if st.button("Show Model Performance", key="show_performance"):
                historical_data = data_manager.get_historical_data(ticker)
                if historical_data is not None:
                    with st.spinner("Analyzing model performance..."):
                        temp_model, X_test, y_test, _ = model_manager.get_or_train_model(ticker, historical_data)
                        st.write(f"Model loaded: {temp_model is not None}")
                        st.write(f"X_test available: {X_test is not None}")
                        st.write(f"y_test available: {y_test is not None}")
                        if temp_model is not None and X_test is not None and y_test is not None:
                            cm, report = model_manager.get_model_performance(ticker, X_test, y_test)                
                            if cm is not None and report is not None:
                                viz.plot_confusion_matrix(cm, ticker)
                    
                                st.write("Classification Report:")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.format(precision=2))

                            else:
                                st.warning("Unable to generate model performance metrics.")
                        else:
                            st.error("Failed to train model for perfomance analysis.")
                else:
                    st.error("Historical data not available for model training.")                    

if __name__ == "__main__":
    main()
