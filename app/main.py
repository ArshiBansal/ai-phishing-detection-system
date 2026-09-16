# app/main.py
"""
PhreshPhish Low-FP Phishing Classifier – Gradio App
"""

import sys
from pathlib import Path

# Make sure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from app.inference import PhishingPredictor


# ------------------------------------------------------------------
# Load the predictor once at startup
# ------------------------------------------------------------------
print("Loading models... this may take a few seconds.")
predictor = PhishingPredictor(
    prefer="transformer",          # "transformer" | "ml" | "ensemble"
    ml_bundle_path="models/exported/ml_bundle.joblib",
    dl_model_dir="models/dl/best_model",
)
print("Models loaded successfully.")


# ------------------------------------------------------------------
# Inference function for Gradio
# ------------------------------------------------------------------
def classify(url: str, html: str):
    if not url or not url.strip():
        return "Please provide a URL.", 0.0, "N/A"

    result = predictor.predict(url=url.strip(), html=html or "")

    label = result["label"].upper()
    prob = result["probability"]
    model_used = result.get("model", "unknown")

    # Friendly message
    if label == "PHISH":
        message = f"⚠️  PHISHING DETECTED (confidence: {prob:.1%})"
    else:
        message = f"✅  Looks benign (confidence: {1-prob:.1%})"

    return message, round(prob, 4), model_used


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
examples = [
    [
        "https://secure-paypal-login.com/verify",
        "<html><body><h1>PayPal Account Verification</h1><p>Please enter your password to continue.</p><form><input type='password' name='pass'></form></body></html>",
    ],
    [
        "https://www.wikipedia.org",
        "<html><head><title>Wikipedia</title></head><body><h1>Wikipedia The Free Encyclopedia</h1></body></html>",
    ],
    [
        "http://192.168.1.1/admin",
        "<html><body>Router Admin Login <input type='password'></body></html>",
    ],
]

demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(
            label="URL",
            placeholder="https://example.com/login",
            lines=1,
        ),
        gr.Textbox(
            label="HTML (optional – paste page source or leave empty)",
            placeholder="<html>...</html>",
            lines=8,
        ),
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Number(label="Phishing Probability"),
        gr.Textbox(label="Model Used"),
    ],
    title="PhreshPhish – Low False-Positive Phishing Classifier",
    description=(
        "Robust phishing webpage detector trained on the large-scale real-world "
        "**PhreshPhish** dataset from Hugging Face. "
        "Optimised for low false-positive rate."
    ),
    examples=examples,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )