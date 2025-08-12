import pandas as pd
import numpy as np
import ta
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models, create models directory if it doesn't exist"""
        os.makedirs('models', exist_ok=True)
        
        for ticker in ['AAPL', 'MSFT', 'INTC', 'IBM']:
            try:
                model = joblib.load(f'models/{ticker}_model.pkl')
                reverse_map = joblib.load(f'models/{ticker}_reverse_map.pkl')
                self.models[ticker] = {
                    'model': model,
                    'reverse_map': reverse_map
                }
                print(f"Model for {ticker} loaded successfully")
            except FileNotFoundError:
                print(f"Model for {ticker} not found - will train on demand")
    
    def get_or_train_model(self, ticker, df):
        """Get existing model or train a new one"""
        if ticker not in self.models:
            print(f"Training new model for {ticker}")
            model, X_test, y_test, reverse_map = self.build_model(df)
            
            if model is not None:
                self.models[ticker] = {
                    'model': model,
                    'reverse_map': reverse_map
                }
                
                joblib.dump(model, f'models/{ticker}_model.pkl')
                joblib.dump(reverse_map, f'models/{ticker}_reverse_map.pkl')
                print(f"Model for {ticker} trained and saved")
                
                return model, X_test, y_test, reverse_map
            else:
                print(f"Failed to train model for {ticker}")
                return None, None, None, None
        else:
            print(f"Using existing model for {ticker}")
            model = self.models[ticker]['model']
            reverse_map = self.models[ticker]['reverse_map']
            
            try:
                processed_df = self.generate_features(df)
                processed_df = self.generate_labels(processed_df)
                
                processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                processed_df.dropna(inplace=True)
                
                features = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10', 'Momentum_5', 
                           'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV']
                X = processed_df[features]
                y = processed_df['Action']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                label_map = {-1: 0, 0: 1, 1: 2}
                y_mapped = y.map(label_map)
                
                _, X_test, _, y_test = train_test_split(
                    X_scaled, y_mapped, test_size=0.2, shuffle=False, random_state=42
                )
                
                return model, X_test, y_test, reverse_map
            except Exception as e:
                print(f"Error generating test data for existing model: {e}")
                return model, None, None, reverse_map
    
    def build_model(self, df):
        """Build and train model with improved parameters"""
        try:
            df = self.generate_features(df)
            df = self.generate_labels(df)
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            if len(df) < 50:  
                print(f"Not enough data for training: {len(df)} rows")
                return None, None, None, None
            
            features = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10', 'Momentum_5', 
                       'Momentum_10', 'Daily_Return', 'Volume_Lag1', 'OBV']
            X = df[features]
            y = df['Action']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            label_map = {-1: 0, 0: 1, 1: 2}
            reverse_map = {v: k for k, v in label_map.items()}
            y_mapped = y.map(label_map)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_mapped, test_size=0.2, shuffle=False, random_state=42
            )
            
            logistic = LogisticRegression(
                max_iter=2000,  
                solver='saga',  
                class_weight='balanced', 
                random_state=42,
                C=0.1  
            )
            
            xgb_clf = XGBClassifier(
                objective='multi:softmax', 
                num_class=3, 
                eval_metric='mlogloss',
                random_state=42,
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100
            )
            
            ensemble = VotingClassifier(
                estimators=[('logistic', logistic), ('xgb', xgb_clf)], 
                voting='soft'
            )
            
            ensemble.fit(X_train, y_train)
            
            return ensemble, X_test, y_test, reverse_map
            
        except Exception as e:
            print(f"Error building model: {e}")
            return None, None, None, None
    
    def predict(self, ticker, features):
        """Make prediction using pre-trained model"""
        if ticker not in self.models:
            print(f"No model available for {ticker}")
            return None
        
        try:
            model = self.models[ticker]['model']
            reverse_map = self.models[ticker]['reverse_map']
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            predicted_class = model.predict(features_scaled)[0]
            predicted_label = reverse_map[predicted_class]
            
            return predicted_label
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_model_performance(self, ticker, X_test, y_test):
        """Get model performance metrics"""
        if ticker not in self.models:
            print(f"No model available for {ticker}")
            return None, None
        
        if X_test is None or y_test is None:
            print(f"Test data not available for {ticker}")
            return None, None
        
        try:
            model = self.models[ticker]['model']
            reverse_map = self.models[ticker]['reverse_map']
            
            y_pred = model.predict(X_test)
            y_test_orig = y_test.map(reverse_map)
            y_pred_orig = pd.Series(y_pred).map(reverse_map)
            
            cm = confusion_matrix(y_test_orig, y_pred_orig, labels=[-1, 0, 1])
            report = classification_report(y_test_orig, y_pred_orig, output_dict=True)
            
            return cm, report
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return None, None
    
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
        
        try:
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['Close'], volume=df['Volume']
            ).on_balance_volume()
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            df['OBV'] = 0
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
                else:
                    df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        
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
