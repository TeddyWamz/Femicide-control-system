GBV Predictor Web App
======================

This repository contains a fine-tuned XLM-RoBERTa model for classifying reports into GBV-related categories and a minimal Flask web application with authentication. Authenticated users can submit a report message and receive the predicted category.

Project structure
-----------------

- `model_training.py` — training script for the classifier.
- `best_gbv_model/` — saved fine-tuned model and tokenizer.
- `webapp/` — Flask app integrating auth and prediction.
  - `app.py` — Flask server, auth routes, and report form.
  - `predictor.py` — loads model once and exposes `predict()`.
  - `templates/` — HTML templates for login/register/report views.

Quick start
-----------

1) Create a virtual environment and install dependencies:

   - Windows (PowerShell)

     - `python -m venv .venv`
     - `.venv\\Scripts\\Activate.ps1`
     - `pip install -r requirements.txt`

2) Ensure the trained model directory exists (default `best_gbv_model/`). If not, run training or copy your fine-tuned model into that folder.

3) Launch the app:

   - Flask dev server (simple): `python webapp/app.py`
   - Or via Flask CLI: set `FLASK_APP=webapp.app:create_app` then `flask run --port 8000`
   - Or via Uvicorn (ASGI wrapper): `uvicorn webapp.app:asgi_app --port 8000`

4) Open the app at `http://localhost:8000`:

   - Register a user, log in, submit a message on the Report page, and view the predicted category and confidence.

API endpoints
-------------

- `GET /healthz` → `{ "status": "ok" }`
- `POST /api/predict` → JSON body `{ "message": "..." }`, returns `{ label, confidence, message }`
- Aliases: `POST /predict`, `POST /classify` (for compatibility with existing clients)

Notes
-----

- The label mapping is kept consistent with training: `0: Physical_violence, 1: sexual_violence, 2: emotional_violence, 3: economic_violence`.
- The predictor loads the model on startup for better performance.
- For production, change `SECRET_KEY`, consider proper user/session management, and run behind a production WSGI server.
