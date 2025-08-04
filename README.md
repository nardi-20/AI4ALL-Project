# AI-Driven Stock Investment Recommendation System
This project was developed during the AI4ALL Ignite Fellowship to explore how artificial intelligence can support smarter financial decision-making. We built and deployed a machine learning system that predicts short-term stock actions (Buy, Hold, or Sell) based on historical stock price data and macroeconomic trends.

Focusing on major tech companies like Apple, Microsoft, Intel, and IBM, the system analyzes daily price patterns alongside indicators such as interest rates, inflation, and unemployment to generate real-time investment recommendations.

## Research Question
Can machine learning models accurately predict daily stock actions (Buy, Hold, Sell) for major tech companies based on historical price trends and macroeconomic indicators?

## Project Overview
- Built ensemble classification models using Logistic Regression and XGBoost
- Engineered 20+ predictive features from stock data and macroeconomic indicators
- Achieved 47% classification accuracy with 100% recall for Hold, and improved macro-averaged F1-score by 25%
- Deployed via Streamlit to deliver real-time predictions and financial visualizations
- Conducted exploratory data analysis to uncover correlations between policy and market behavior

## Technologies Used
- Python (pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn)
- Streamlit (for interactive dashboard and deployment)
- Jupyter Notebook (for EDA and model development)

## Datasets
Located in the /AI4ALL Project Datasets directory, our data sources include:

- Historical stock prices: AAPL, MSFT, INTC, IBM
- U.S. economic indicators: CPI, GDP, Unemployment, Federal Funds Rate
- Global inflation rates by country

## Methods
- Supervised Learning (Logistic Regression, XGBoost)
- Feature Engineering (Lagged prices, Moving averages, Rolling correlations)
- Exploratory Data Analysis and Visualization
- Correlation analysis between stock performance and economic trends

## Key Features
- Buy/Hold/Sell predictions using ensemble machine learning models
- Rolling and lagged correlation visualizations between stocks and economic indicators
- Normalized and merged datasets prepared for model training
- Interactive Streamlit app for real-time stock recommendations
- Confusion matrices and classification reports to evaluate model performance

## Installation & Usage
1) Clone this repository:

   > git clone [https://github.com/your-username/ai4all-stock-recommendation.git]
   >
   > cd ai4all-stock-recommendation
3) Install required packages:

   > pip install -r requirements.txt
5) Run the Streamlit app:

   > streamlit run final_project.py

Ensure the /AI4ALL Project Datasets folder is in the root directory. The app will load and display real-time stock recommendations based on preprocessed and merged datasets.

## Example Predictions
Predictions are categorized as Buy, Hold, or Sell. Additional charts display:

- Stock price trends
- Economic indicator overlays
- Visual insights from the model's decision-making process

## Potential Applications
- Real-time decision support for retail investment platforms
- Financial insight tools for new investors
- Market insight visualizations for beginner investors
- Educational dashboards for finance learners

## Contributing
Contributions are welcome! To get started:

1) Fork the repo
2) Create a new branch (git checkout -b feature-name)
3) Commit your changes (git commit -m 'Add new feature')
4) Push to the branch (git push origin feature-name)
5) Open a Pull Request
   
Please follow PEP8 standards for Python and include comments where necessary.

## Acknowledgments
This project was developed as part of the AI4ALL Ignite Fellowship. Special thanks to our mentors and program staff for their guidance and feedback. Additional thanks to the providers of our stock and economic data, including Yahoo Finance, FRED, and Kaggle.
