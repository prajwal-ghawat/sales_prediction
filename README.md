# 📊 Sales Prediction using Linear Regression

## Project Overview
This project predicts product sales based on advertising expenditure using a Machine Learning approach. A Linear Regression model is trained on historical advertising data to understand and estimate the relationship between advertising channels and sales.

## Objective
- Analyze the impact of advertising on sales
- Build a predictive model using Linear Regression
- Evaluate model performance using standard metrics

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

## Project Structure
Sales-Prediction/
├── sales_prediction.py  
├── Advertising.csv  
├── images/  
│   ├── feature_correlation.png  
│   └── actual_vs_predicted_sales.png  
├── models/  
│   └── linear_regression_sales.pkl  
└── README.md  

## Dataset Description
Advertising.csv contains advertising budgets and corresponding sales values.

Columns:
- TV – Advertising spend on TV
- Radio – Advertising spend on Radio
- Newspaper – Advertising spend on Newspaper
- Sales – Product sales (target variable)

## Exploratory Data Analysis
- Dataset preview and summary statistics
- Correlation analysis using heatmap
- Relationship analysis between advertising channels and sales

## Model Used
Linear Regression model is used to predict continuous sales values based on advertising spend.

## Model Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Visualizations
- Correlation heatmap of features
- Actual vs Predicted sales plot

## How to Run the Project
1. Navigate to the project directory  
2. Run the Python script using:  
   python sales_prediction.py

## Model Saving
The trained model is saved for future use in the models folder.

## Results
The model demonstrates good prediction accuracy and effectively captures the relationship between advertising expenditure and sales.

## Conclusion
This project shows how Linear Regression can be applied to predict sales using advertising data and serves as a foundation for more advanced regression models.

## Future Enhancements
- Implement advanced regression models
- Add real-time prediction input
- Improve feature engineering

## Author
Prajwal Ghawat

## License
Educational use only
