# ChurnIQ — Customer Churn Analysis & Prediction

An end-to-end customer churn project that combines exploratory data analysis, a Power BI dashboard, and a live FastAPI-powered prediction engine called **ChurnIQ**. The goal is to understand *why* customers leave and to predict *which* customers are likely to churn next, so retention efforts can be targeted before it's too late.

---

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Prediction Engine (ChurnIQ)](#prediction-engine-churniq)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Running the Analysis](#running-the-analysis)
- [Running the Prediction App](#running-the-prediction-app)
- [API Reference](#api-reference)
- [Model](#model)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Customer churn — when a subscriber stops doing business with a company — is one of the most expensive problems a subscription or telecom business can face. This project tackles it in three layers:

1. **Data Analysis (SQL + Jupyter Notebook)** — clean and explore the raw customer data to surface patterns behind churn.
2. **Business Intelligence (Power BI)** — a summary dashboard that lets stakeholders slice churn by demographics, contract type, services, geography, and reason for leaving.
3. **Machine Learning (FastAPI app)** — a trained model served behind a simple web interface, so a support or retention team can enter a customer's profile and get an instant churn-risk prediction.

## Dashboard

The Power BI dashboard (`Customer-Churn-Analysis_Dashboard.pdf`) gives a single-page summary view of churn across the customer base:

- **Headline metrics** — total customers, new joiners, total churn, and overall churn rate.
- **Demographics** — churn split by gender and by age group.
- **Account info** — churn rate by payment method, contract type, and tenure group.
- **Geographic** — top states/regions by churn rate.
- **Churn distribution** — churn broken down by category (e.g. competitor, price, dissatisfaction, attitude, other).
- **Services used** — churn rate by individual service (internet, phone, streaming, security add-ons, etc.), shown as a Yes/No breakdown so you can see which add-ons correlate with retention.
- **Filters** — interactive slicers for monthly charge range and marital status so the whole report can be re-cut on demand.

To explore it yourself, open the PDF in this repo, or recreate the live version by opening the underlying `.pbix` file (if included) in Power BI Desktop.

## Prediction Engine (ChurnIQ)

ChurnIQ is the interactive front end for the trained churn model. It's a lightweight web app (FastAPI + HTML/CSS/JS) where a user fills out a customer's profile across three tabs and gets a prediction back instantly.

- **Demographics tab** — gender, senior citizen status, partner, dependents, tenure (months).
- **Services tab** — phone, internet, and add-on services the customer subscribes to.
- **Billing tab** — contract type, payment method, monthly/total charges.
- **Result panel** — once a profile is submitted, the panel on the right displays the model's churn prediction and probability instead of the placeholder prompt.
- **Status indicator** — an "API Online" badge confirms the backend is reachable, with a **Reset** button to clear the form and start over.

## Project Structure

```
Customer_Churn_Analysis/
│
├── models/                              # Serialized ML model(s) and any preprocessing objects
├── templates/
│   └── index.html                       # Front-end UI for the ChurnIQ prediction app
├── __pycache__/                         # Python bytecode cache (auto-generated, can be ignored)
│
├── app.py                               # Application entry point / core app logic
├── fastapi_app.py                       # FastAPI app exposing the prediction endpoint(s)
├── requirements.txt                     # Python dependencies
│
├── Customer_Data.csv                    # Raw/cleaned customer dataset
├── SQLQuery5.sql                        # SQL used for data cleaning/exploration
├── Customer_Churn_Analysis.ipynb        # Notebook: EDA, feature engineering, model training
└── Customer-Churn-Analysis_Dashboard.pdf # Exported Power BI summary dashboard
```

> Note: adjust the tree above if your actual file names or folders differ slightly — keep this section in sync with the repo as it evolves.

## Tech Stack

| Layer            | Tools |
|-------------------|-------|
| Data storage/query | SQL (`SQLQuery5.sql`), CSV |
| Analysis & modeling | Python, Jupyter Notebook, pandas, scikit-learn |
| Visualization (BI) | Power BI |
| Backend API        | FastAPI, Uvicorn |
| Frontend           | HTML/CSS/JavaScript (`templates/index.html`) |
| Model serving       | Pickled/serialized model in `models/` |

## Dataset

`Customer_Data.csv` contains one row per customer with fields typically including:

- **Demographics** — gender, senior citizen flag, partner, dependents
- **Account details** — tenure (months), contract type, paperless billing, payment method
- **Services** — phone service, multiple lines, internet service, online security/backup, device protection, tech support, streaming TV/movies/music
- **Charges** — monthly charges, total charges
- **Target** — churn label (and, where available, churn category/reason)

Update this section with the exact column names and any data dictionary notes specific to your CSV.

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional) Power BI Desktop, to open/edit the dashboard source file

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Customer_Churn_Analysis.git
cd Customer_Churn_Analysis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

Open the notebook to walk through data cleaning, exploratory analysis, feature engineering, and model training:

```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```

If you want to reproduce the SQL-side exploration, run `SQLQuery5.sql` against your database of choice (e.g. SQL Server, PostgreSQL, MySQL) after loading `Customer_Data.csv` into a table.

## Running the Prediction App

Start the FastAPI server:

```bash
uvicorn fastapi_app:app --reload
```

Then open your browser to:

```
http://127.0.0.1:8000
```

This loads the ChurnIQ interface (`templates/index.html`). Fill in the Demographics, Services, and Billing tabs, then submit to see the churn prediction populate in the results panel.

## API Reference

Once the server is running, FastAPI's interactive docs are available at:

```
http://127.0.0.1:8000/docs
```

Typical endpoints to document here (fill in with your actual routes):

| Method | Endpoint     | Description                                  |
|--------|--------------|-----------------------------------------------|
| GET    | `/`          | Serves the ChurnIQ web interface               |
| POST   | `/predict`   | Accepts a customer profile, returns churn prediction + probability |
| GET    | `/health`    | Health check used by the "API Online" indicator |

## Model

The model artifact(s) live in `models/`. Document here:

- Algorithm used (e.g. Logistic Regression, Random Forest, XGBoost)
- Key features and any preprocessing/encoding steps
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) on the test set
- How the model was exported (e.g. `pickle`, `joblib`) and how `fastapi_app.py` loads it

## Future Improvements

- Add authentication to the prediction API
- Containerize the app with Docker for easier deployment
- Add automated retraining as new customer data arrives
- Expand the dashboard with cohort/retention curves over time
- Add unit tests for the FastAPI endpoints

## License

Specify your license here (e.g. MIT, Apache 2.0) or remove this section if the repo is private/unlicensed.
