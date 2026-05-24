# ⛽ VicFuelCast — Suburb-Level Fuel Price Prediction (VIC)

## Group: CL04 — G04

##Deployed web app here [FuelForcastVic](https://vicfuelcos40007.web.app/)

### Team Members
- **Soad Yusuf** — 105406263  
- **Amiru Adhikari** — 105522350  
- **Gia** — 105028560  
- **Gavin Fernando** — 104697341  

---

# 📌 Project Overview

VicFuelCast is an AI-powered fuel price prediction system designed to forecast fuel prices at a suburb level across Victoria, Australia.

The project combines machine learning, data engineering, and MLOps practices to help users make smarter and more cost-effective refuelling decisions.

By analysing historical and live fuel pricing data, the system aims to provide accurate suburb-level fuel price forecasts and support better planning for Victorian drivers.

---

# 🎯 Vision

The fuel market in Victoria is highly unstable, creating financial pressure for families and individuals.

VicFuelCast aims to reduce this uncertainty by providing suburb-level fuel price predictions, giving Victorians better visibility into future fuel trends and helping them make informed refuelling decisions.

---

# 🚀 Project Goal

Develop a close-to-accurate machine learning model capable of predicting fuel prices across Victorian suburbs using historical and live fuel pricing data.

---

# 📊 Data Sources

## Primary Source — VIC Servo (Service Victoria)

- Daily fuel pricing data from fuel stations across Victoria
- Data includes multiple fuel types and suburb-level pricing information
- Retrieved through authorised API access using a Service Victoria API key
- Data availability includes a 24-hour delay as defined by VIC Servo

---

## Secondary Source — ACCC (Australian Competition and Consumer Commission)

- Weekly published fuel pricing reports
- Historical pricing data used for trend analysis and model validation
- Data collected using automated scraping processes
- Scraper executes 24 hours after report publication

---

# 🧠 Core Features

- Suburb-level fuel price forecasting
- Historical and live data integration
- Machine learning-based prediction pipeline
- Automated preprocessing and retraining workflow
- GitHub Actions CI/CD pipeline
- Monitoring and artefact generation
- Forecast visualisation and reporting

---

# ⚙️ Technologies Used

## Programming & ML
- Python
- TensorFlow
- Scikit-learn
- Pandas
- NumPy

## MLOps & Automation
- GitHub Actions
- DVC
- Git

## Visualisation
- Matplotlib

---

# 🔄 MLOps Pipeline

The project includes a complete MLOps workflow:

1. Data ingestion
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Monitoring and drift detection
7. Automated retraining
8. Artefact generation and storage

---

# 📁 Repository Structure

```text
.github/workflows/   → GitHub Actions workflows
code/                → Model and pipeline scripts
data/                → Datasets
models/              → Saved trained models
reports/             → Generated reports and outputs
model.py             → Main pipeline execution script
requirements.txt     → Python dependencies
README.md            → Project documentation
