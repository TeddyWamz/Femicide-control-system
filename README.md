Femicide reporting Web App
A machine learning-powered web application for classifying Gender-Based Violence (GBV) and femicide-related reports. This project combines a fine-tuned XLM-RoBERTa model with a Flask web interface, enabling authenticated users to submit reports and receive instant, explainable predictions.

Table of Contents
Background & Motivation
Project Objectives
System Architecture
Data Sources & Preprocessing
Model Training & Evaluation
Web Application Features
API Documentation
Deployment Guide
Contribution Guidelines
Troubleshooting & FAQ
License
Background & Motivation
Gender-Based Violence (GBV) and femicide are critical issues worldwide. Timely identification and categorization of reports can help organizations respond more effectively. This project leverages NLP and deep learning to automate the classification of GBV-related reports, supporting both research and intervention efforts.

Project Objectives
Automate classification of GBV and femicide reports into actionable categories.
Provide a secure, user-friendly web interface for report submission and feedback.
Enable rapid deployment and easy retraining with new data.
Support explainability and transparency in model predictions.
System Architecture
Backend: Python 3.13, Flask 3.1.2, scikit-learn, transformers (Hugging Face), torch, psycopg, pandas.
Frontend: HTML (Jinja2 templates), Bootstrap (optional for styling).
Database: PostgreSQL (for user authentication and report storage).
Model: XLM-RoBERTa fine-tuned for multi-label GBV classification.
Data Sources & Preprocessing
Datasets:
gbv_cleaned_dataset.csv, gbv_comprehensive_dataset.csv, final_training_data.csv (see repo for samples)
Preprocessing Steps:
Text normalization, label mapping, deduplication, and train/test split.
See augment_data.py and data_inspection.py for scripts.
Model Training & Evaluation
Script: model_training.py
Model: XLM-RoBERTa (Hugging Face Transformers)
Metrics: Precision, Recall, F1-score (macro & per-class)
Artifacts:
best_gbv_model/ (PyTorch weights, tokenizer, config)
potential_mislabeled.csv (for error analysis)
Reproducibility:
All random seeds fixed, requirements pinned in requirements.txt
Web Application Features
User registration and login (Flask auth)
Secure report submission form
Real-time prediction with model confidence
Admin dashboard (optional, for reviewing reports)
Email notifications (see mailer.py)
API Documentation
GET /healthz: Health check, returns { "status": "ok" }
POST /api/predict: Predicts GBV category
Request: { "message": "..." }
Response: { "label": "...", "confidence": 0.95, "message": "..." }
Aliases: /predict, /classify
Authentication: Session-based (Flask-Login)
Error Handling: Returns HTTP 400/500 with JSON error messages
Deployment Guide
Clone the repository:
git clone https://github.com/TeddyWamz/Femicide-control-system.git
cd Femicide-control-system
Set up Python environment:
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
Prepare the model:
Ensure best_gbv_model/ exists with model and tokenizer files.
To retrain, run python model_training.py.
Configure environment variables:
Set SECRET_KEY, database URI, and mail settings in .env or directly in app.py.
Run the app:
python webapp/app.py
# or
flask run --port 8000
Access the app:
Open http://localhost:8000 in your browser.
Contribution Guidelines
Fork the repo and create a feature branch.
Write clear commit messages and document your code.
Open a pull request with a detailed description.
Follow PEP8 and project code style.
For major changes, open an issue first to discuss.
Troubleshooting & FAQ
Model not found? Ensure best_gbv_model/ is present and paths are correct.
Dependency errors? Double-check Python version and run pip install -r requirements.txt.
Database issues? Confirm PostgreSQL is running and credentials are correct.
App not starting? Check for missing environment variables or port conflicts.
License
This project is licensed under the MIT License. See LICENSE for details.
