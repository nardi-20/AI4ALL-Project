import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        for ticker in ['AAPL', 'MSFT', 'INTC', 'IBM']:
            try:
                model = joblib.load(f'models/{ticker}_model.pkl')
                reverse_map = joblib.load(f'models/{ticker}_reverse_map.pkl')
                self.models[ticker] = {
                    'model': model,
                    'reverse_map': reverse_map
                }
            except FileNotFoundError:
                print(f"Model for {ticker} not found")
    
    def build_model(self, df):
        """Build and train model"""
        df = self.generate_features(df)
        df = self.generate_labels(df)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        features = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10', 'Momentum_5', 
                   'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV']
        X = df[features]
        y = df['Action']
        
        label_map = {-1: 0, 0: 1, 1: 2}
        reverse_map = {v: k for k, v in label_map.items()}
        y_mapped = y.map(label_map)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_mapped, test_size=0.2, shuffle=False
        )
        
        logistic = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        xgb_clf = XGBClassifier(
            objective='multi:softmax', num_class=3, 
            eval_metric='mlogloss', use_label_encoder=False, random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[('logistic', logistic), ('xgb', xgb_clf)], 
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        
        return ensemble, X_test, y_test, reverse_map
    
    def predict(self, ticker, features):
        """Make prediction using pre-trained model"""
        if ticker not in self.models:
            return None
        
        model = self.models[ticker]['model']
        reverse_map = self.models[ticker]['reverse_map']
        
        predicted_class = model.predict(features)[0]
        predicted_label = reverse_map[predicted_class]
        
        return predicted_label
    
    def get_model_performance(self, ticker, X_test, y_test):
        """Get model performance metrics"""
        if ticker not in self.models:
            return None
        
        model = self.models[ticker]['model']
        reverse_map = self.models[ticker]['reverse_map']
        
        y_pred = model.predict(X_test)
        y_test_orig = y_test.map(reverse_map)
        y_pred_orig = pd.Series(y_pred).map(reverse_map)
        
        cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])
        report = classification_report(y_test_orig, y_pred_orig, output_dict=True)
        
        return cm, report
    
    def generate_features(self, df):
        """Generate features (duplicate from DataManager for independence)"""
        df = df.copy()
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volume_Lag1'] = df['Volume'].shift(1)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        
        return df
    
    def generate_labels(self, df, column='Close', days_ahead=5, threshold=0.005):
        """Generate labels (duplicate from DataManager for independence)"""
        df = df.copy()
        df['FuturePrice'] = df[column].shift(-days_ahead)
        df['Return'] = (df['FuturePrice'] - df[column]) / df[column]
        df['Action'] = df['Return'].apply(
            lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
        )
        return df

