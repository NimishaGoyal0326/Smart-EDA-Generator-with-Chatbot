# 🧠 Smart EDA Chatbot

A **self-contained, AI-powered Exploratory Data Analysis chatbot** that trains ML models locally on any CSV/Excel dataset. No external APIs required — everything runs on your machine.

## ✨ Features

- **💬 Natural Language Chat** — Ask any question about your data in plain English
- **🧠 Self-Trained ML Models** — Trains Logistic/Linear Regression, Random Forest, and Gradient Boosting locally
- **📊 Full Auto-EDA** — Statistics, missing values, correlations, outliers, distributions, categorical analysis
- **🔮 Predictions** — Enter feature values and get predictions from trained models
- **📈 Interactive Visualizations** — Histograms, box plots, scatter plots, correlation heatmaps (Plotly)

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔧 Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | **Streamlit** | Interactive dashboard |
| Data Processing | **Pandas, NumPy** | Data manipulation |
| ML Training | **Scikit-learn** | Classification & regression models |
| Statistics | **SciPy** | Statistical tests & distributions |
| Visualization | **Plotly** | Interactive charts |

## 📁 Project Structure

```
EDA CHATBOT/
├── app.py              # Main Streamlit application
├── ai_chatbot.py       # SmartEDA engine (ML training + query answering)
├── eda_utils.py        # Visualization utilities (Plotly charts)
├── requirements.txt    # Python dependencies
└── .streamlit/         # Streamlit configuration
```

## 💡 What You Can Ask

- **Statistics:** "Show summary statistics", "What is the average income?"
- **Missing Data:** "Which columns have missing values?"
- **Correlations:** "What are the strongest correlations?"
- **Outliers:** "Show outlier analysis"
- **Distributions:** "What is the distribution of age?"
- **Predictions:** "What does the model predict?"
- **Probability:** "What is the probability of default?"
- **Model Info:** "What is the model accuracy?", "Which features are most important?"
- **Column Stats:** "Max salary?", "Median age?"
