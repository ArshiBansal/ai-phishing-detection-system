# app/ui.py
"""
Gradio UI definition for the PhreshPhish classifier.
Separated from main.py so it can be reused or tested independently.
"""

import gradio as gr
from typing import Callable


def create_ui(classify_fn: Callable) -> gr.Blocks:
    """
    Build and return the Gradio Blocks interface.

    Parameters
    ----------
    classify_fn : callable
        Function with signature (url: str, html: str) -> (message, probability, model_used)
    """

    examples = [
        [
            "https://secure-paypal-login.com/verify",
            "<html><body><h1>PayPal Account Verification</h1>"
            "<p>Please enter your password to continue.</p>"
            "<form><input type='password' name='pass'></form></body></html>",
        ],
        [
            "https://www.wikipedia.org",
            "<html><head><title>Wikipedia</title></head>"
            "<body><h1>Wikipedia The Free Encyclopedia</h1></body></html>",
        ],
        [
            "http://192.168.1.1/admin",
            "<html><body>Router Admin Login <input type='password'></body></html>",
        ],
        [
            "https://accounts.google.com",
            "<html><head><title>Sign in – Google Accounts</title></head>"
            "<body>Sign in with your Google Account</body></html>",
        ],
    ]

    with gr.Blocks(
        title="PhreshPhish Classifier",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 900px !important; }
        .result-box { font-size: 1.15rem; font-weight: 600; }
        """
    ) as demo:

        gr.Markdown(
            """
            # PhreshPhish – Low False-Positive Phishing Classifier
            Robust detector trained on the large-scale real-world **PhreshPhish** dataset  
            (Hugging Face). Optimised for **high precision / low false-positive rate**.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    label="URL",
                    placeholder="https://example.com/login",
                    lines=1,
                    max_lines=2,
                )
                html_input = gr.Textbox(
                    label="HTML source (optional)",
                    placeholder="Paste page source here, or leave empty to use URL only",
                    lines=10,
                    max_lines=20,
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=2):
                result_box = gr.Textbox(
                    label="Result",
                    interactive=False,
                    elem_classes=["result-box"],
                )
                prob_box = gr.Number(
                    label="Phishing Probability",
                    precision=4,
                    interactive=False,
                )
                model_box = gr.Textbox(
                    label="Model Used",
                    interactive=False,
                )

        gr.Examples(
            examples=examples,
            inputs=[url_input, html_input],
            label="Try these examples",
        )

        gr.Markdown(
            """
            ---
            **Notes**
            - Higher probability → higher chance the page is phishing
            - Decision threshold is tuned for low false-positive rate
            - For best results, paste the full HTML source when available
            """
        )

        # Wire the button
        submit_btn.click(
            fn=classify_fn,
            inputs=[url_input, html_input],
            outputs=[result_box, prob_box, model_box],
        )

        # Also allow Enter key on URL field
        url_input.submit(
            fn=classify_fn,
            inputs=[url_input, html_input],
            outputs=[result_box, prob_box, model_box],
        )

    return demo