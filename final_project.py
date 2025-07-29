import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import ta

appleDF = pd.read_csv('AI4ALL Project Datasets/AAPL.csv')
intelDF = pd.read_csv('AI4ALL Project Datasets/INTC.csv')
msftDF = pd.read_csv('AI4ALL Project Datasets/MSFT.csv')
ibmDF = pd.read_csv('AI4ALL Project Datasets/IBM.csv')
sp500DF = pd.read_csv('AI4ALL Project Datasets/GSPC.csv')
interestRateDF = pd.read_csv('AI4ALL Project Datasets/federalReserveInterestRates.csv')

for df in [appleDF, intelDF, msftDF, ibmDF, sp500DF]:
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

interestRateDF = interestRateDF.interpolate()
interestRateDF['Date'] = pd.to_datetime(interestRateDF[['Year', 'Month', 'Day']])

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def plot_time_series(df, x_col, y_col, label, title, color='blue'):
    plt.figure(figsize=(14, 6))
    plt.plot(df[x_col], df[y_col], label=label, color=color)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_dual_series(df, x_col, y1_col, y2_col, label1, label2, color1='blue', color2='red', title=''):
    plt.figure(figsize=(14, 6))
    plt.plot(df[x_col], df[y1_col], label=label1, color=color1)
    plt.plot(df[x_col], df[y2_col], label=label2, color=color2)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_rolling_correlation(df1, df2, window, title):
    merged = pd.merge_asof(df1[['Date', 'Open']], df2[['Date', 'Effective Federal Funds Rate']], on='Date')
    merged['RollingCorr'] = merged['Open'].rolling(window=window).corr(merged['Effective Federal Funds Rate'])
    plt.figure(figsize=(14, 6))
    plt.plot(merged['Date'], merged['RollingCorr'], color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Rolling Correlation')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return merged

def plot_heatmap(corr_matrix, title=''):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def compare_to_sp500(stockDF, name):
    stockDF = stockDF.sort_values('Date')
    merged = pd.merge_asof(stockDF[['Date', 'Open']], sp500DF[['Date', 'Close']], on='Date').dropna()
    merged['Stock_norm'] = normalize(merged['Open'])
    merged['SP500_norm'] = normalize(merged['Close'])
    plt.figure(figsize=(14, 6))
    plt.plot(merged['Date'], merged['Stock_norm'], label=f'{name} (Normalized)', color='blue')
    plt.plot(merged['Date'], merged['SP500_norm'], label='S&P 500 (Normalized)', color='gray')
    plt.title(f'{name} vs S&P 500 (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    correlation = merged['Stock_norm'].corr(merged['SP500_norm'])
    st.write(f"Correlation between {name} and S&P 500: {correlation:.3f}")

def correlation_matrix(df_dict, price_type='Open'):
    merged = None
    for name, df in df_dict.items():
        temp = df[['Date', price_type]].rename(columns={price_type: f"{name}_{price_type}"})
        merged = temp if merged is None else pd.merge(merged, temp, on='Date', how='outer')
    merged.sort_values('Date', inplace=True)
    merged.interpolate(method='linear', inplace=True)
    merged.dropna(inplace=True)
    return merged.drop(columns='Date').corr()

def generate_labels(df, column='Close', days_ahead=5, threshold=0.005):
    df['FuturePrice'] = df[column].shift(-days_ahead)
    df['Return'] = (df['FuturePrice'] - df[column]) / df[column]
    df['Action'] = df['Return'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    return df

def generate_features(df):
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volume_Lag1'] = df['Volume'].shift(1)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    return df

def build_model(df):
    df = generate_labels(df)
    df = generate_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    features = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10', 'Momentum_5', 'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV']
    X = df[features]
    y = df['Action']
    label_map = {-1: 0, 0: 1, 1: 2}
    reverse_map = {v: k for k, v in label_map.items()}
    y_mapped = y.map(label_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, shuffle=False)
    logistic = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    xgb_clf = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    ensemble = VotingClassifier(estimators=[('logistic', logistic), ('xgb', xgb_clf)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble, X_test, y_test, reverse_map

def test_model(model, X_test, y_test, reverse_map, stock_name):
    y_pred = model.predict(X_test)
    y_test_orig = y_test.map(reverse_map)
    y_pred_orig = pd.Series(y_pred).map(reverse_map)
    cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])
    labels = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{stock_name} — Ensemble Confusion Matrix')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    report_dict = classification_report(y_test_orig, y_pred_orig, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader(f"Classification Report for {stock_name}")
    st.dataframe(report_df.style.format(precision=2))

st.title("Stock Analysis & Investment Prediction")

stock_options = {
    'Apple (AAPL)': appleDF,
    'Microsoft (MSFT)': msftDF,
    'Intel (INTC)': intelDF,
    'IBM': ibmDF
}
stock_name = st.selectbox("Choose a company:", list(stock_options.keys()))
df = stock_options[stock_name]

if st.checkbox("Show raw data"):
    st.write(df.tail())

if st.button("Plot Time Series"):
    plot_time_series(df, x_col='Date', y_col='Close', label=f'{stock_name} Close Price', title=f'{stock_name} Closing Prices Over Time')

if st.button("Compare to S&P 500"):
    compare_to_sp500(df, stock_name)

if st.button("Train Ensemble Model"):
    st.write("Training model...")
    model, X_test, y_test, reverse_map = build_model(df)
    test_model(model, X_test, y_test, reverse_map, stock_name)

    # Save model in session state so it's accessible by other buttons
    st.session_state["model"] = model
    st.session_state["reverse_map"] = reverse_map
    st.success("Model trained and saved!")


if st.button("Predict Latest Movement"):
    if "model" not in st.session_state:
        st.error("Please train the model first.")
    else:
        model = st.session_state["model"]
        reverse_map = st.session_state["reverse_map"]
        processed_df = generate_features(df)
        last_features = processed_df.tail(1)[[
            'Close_Lag1', 'Close_Lag2', 'MA5', 'MA10',
            'Momentum_5', 'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV'
        ]]

        if last_features.isnull().values.any():
            st.warning("Not enough recent data for prediction.")
        else:
            predicted_class = model.predict(last_features)[0]
            predicted_label = reverse_map[predicted_class]
            action_map = {-1: "Sell", 0: "Hold", 1: "Buy"}

            # Show as a nice table
            recommendation_df = pd.DataFrame({
                'Stock': [stock_name],
                'Date': [df['Date'].iloc[-1]],
                'Recommended Action': [action_map[predicted_label]]
            })
            st.subheader("Investment Recommendation")
            st.dataframe(recommendation_df)
