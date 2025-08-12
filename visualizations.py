import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class StockVisualizations:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_time_series(self, df, x_col, y_col, label, title, color='blue'):
        """Plotting of time series data"""
        plt.figure(figsize=(14, 6))
        plt.plot(df[x_col], df[y_col], label=label, color=color)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    def plot_dual_series(self, df, x_col, y1_col, y2_col, label1, label2, color1='blue', color2='red', title=''):
        """Plotting of two series on the same chart"""
        plt.figure(figsize=(14, 6))
        plt.plot(df[x_col], y1_col, label=label1, color=color1)
        plt.plot(df[x_col], y2_col, label=label2, color=color2)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    def plot_heatmap(self, corr_matrix, title=''):
        """Plotting of the correlation heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    def plot_rolling_correlation(self, df1, df2, window, title):
        """Plotting of the rolling correlation between two series"""
        merged = pd.merge_asof(df1[['Date', 'Close']], df2[['Date', 'Effective Federal Funds Rate']], on='Date')
        merged['RollingCorr'] = merged['Close'].rolling(window=window).corr(merged['Effective Federal Funds Rate'])
        
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
    
    def compare_to_sp500(self, stockDF, sp500DF, name):
        """Comparsion of  stock performance to S&P 500"""
        stockDF = stockDF.sort_values('Date')
        merged = pd.merge_asof(stockDF[['Date', 'Close']], sp500DF[['Date', 'Close']], on='Date').dropna()
        
        merged['Stock_norm'] = (merged['Close_x'] - merged['Close_x'].min()) / (merged['Close_x'].max() - merged['Close_x'].min())
        merged['SP500_norm'] = (merged['Close_y'] - merged['Close_y'].min()) / (merged['Close_y'].max() - merged['Close_y'].min())
        
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
        
        return correlation
    
    def plot_confusion_matrix(self, cm, stock_name):
        """matrix for model evaluation"""
        labels = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{stock_name} — Ensemble Confusion Matrix')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    def plot_feature_importance(self, feature_names, importance, title=''):
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        plt.bar(range(len(feature_names)), importance[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
        plt.title(title)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()