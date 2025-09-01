This repository provides a proof-of-concept tool that allows non-coders (e.g., systems engineers) to train, validate, and deploy supervised machine learning (ML) models for use in MBSE (Model-Based Systems Engineering) workflows.
The app uses a Tkinter GUI for model selection and training, leverages an LLM (ChatGPT) to auto-generate model training code, and provides a Flask server for deployment so external tools (e.g., MagicDraw/Cameo, Postman) can interact with trained models.

Installation Instructions
1.  Install all dependencies: pip install -r requirements.txt
2. Get an OpenAI API key from https://platform.openai.com/
3. Open functions.py in this repo.
4. Replace the placeholder with your own key.

Usage Instructions

1. Start the GUI - python gui.py

The GUI allows you to:
I. Select a model (XGBoost, Linear Regression, …)
II. Set hyperparameters
III. Upload a CSV dataset
IV. Train and validate models (RMSE, R², NRMSE shown)
V. Save the trained model for deployment
VI. Deploy the model to a running server

2. Start the Flask Server

I. Run the provided server file: python ml_model_server.py
II. The server exposes three endpoints:
/deploy-model → Upload a trained model (.pkl/.joblib)
/predict → Send JSON input and receive predictions
/status → Check if a model is deployed


Current version supports supervised regression models on flat feature–target datasets (e.g., XGBoost, Linear Regression).
Time-series models (e.g., LSTM) are not supported due to their sequential input requirements.
Classification model support is planned as future work.
