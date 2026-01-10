**🍽️ Zomato Restaurant Rating Prediction**

This project analyzes Zomato restaurant reviews and metadata to predict restaurant ratings using machine learning. It demonstrates an end-to-end data science pipeline — from raw data processing and feature engineering to model deployment as a live web application.

Live App: https://zomato-rating-predictions.streamlit.app/

---

**Project Highlights**

* Data cleaning and preprocessing of review and restaurant metadata

* Feature engineering (textual, numerical, and categorical features)

* Training and comparison of multiple ML models (Random Forest, Linear Regression, SVR)

* Model evaluation using RMSE and R² metrics

* Selection of the best-performing model based on accuracy and stability

* Deployment of the final model as a Streamlit web application
  
---

**Machine Learning Workflow**

* Data ingestion and cleaning

* Exploratory Data Analysis (EDA) and visualization

* Feature engineering and selection

* Model training and evaluation

* Model serialization (pickle)

* Web application deployment using Streamlit

---

**Repository Structure**
```
zomato-analysis/
├── data/
│   └── zomato.csv                     # Raw dataset
├── notebooks/
│   └── Zomato_Analysis.ipynb          # Main analysis notebook
├── reports/
│   └── zomato_report.pdf              # Final report (optional)
├── deployment/
│   ├── app.py                         # Streamlit app
│   ├── random_forest_model.pkl        # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   ├── feature_columns.pkl            # Feature list
│   ├── cuisine_features.pkl           # Cuisine encodings
│   ├── collection_features.pkl        # Collection encodings
│   └── requirements.txt               # Dependencies
└── README.md
```

---

**Model Evaluation**

* The Random Forest Regressor was selected as the final model based on:

* Lowest RMSE compared to other models

* Better generalization performance

* Stability against outliers and noisy inputs

---

**Business Value**

* Helps users estimate expected restaurant ratings before visiting

* Can support restaurant recommendation systems

* Helps platforms detect anomalous or suspicious ratings

* Provides insights into factors influencing customer satisfaction

---

**Author**
Developed by Avinash Singh
B.Tech CSE (AI), AKTU
Interested in Data Science, Machine Learning, and applied analytics

---

**License**
This project is for educational and research purposes.
