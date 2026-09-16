<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=4000&pause=800&color=3B82F6&center=true&vCenter=true&width=700&lines=PhreshPhish+Classifier;Low+False-Positive+Phishing+Detection;ML+%2B+DL+%2B+NLP" alt="Typing SVG" />
</p>

<p align="center">
  <strong>Robust phishing webpage classifier</strong> trained on the large-scale real-world <a href="https://huggingface.co/datasets/phreshphish/phreshphish">PhreshPhish</a> dataset.<br>
  Optimised for <b>high precision</b> and <b>low false-positive rate</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Dataset-PhreshPhish-orange" alt="Dataset"/>
  <img src="https://img.shields.io/badge/App-Gradio-red" alt="Gradio"/>
</p>

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Overview" alt="Overview"/>

Phishing attacks remain a major cybersecurity threat. This project builds a **low-false-positive** phishing webpage classifier using:

- **Classical ML** (XGBoost + TF-IDF + URL features)
- **Deep Learning / NLP** (Transformer fine-tuning)
- **Large-scale real-world data** from Hugging Face (`phreshphish/phreshphish`)

The system is designed for realistic evaluation (low base rates) and prioritises **Precision @ high Recall**.

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Key+Features" alt="Key Features"/>

- Streaming load of PhreshPhish (no full 36 GB download required)
- Memory-safe HTML cleaning & feature extraction
- Strong classical baseline (XGBoost)
- Transformer model fine-tuned for phishing detection
- Optional ensemble for even lower false positives
- Gradio web app for interactive inference
- Evaluation focused on Average Precision & Precision@Recall
- Clean modular codebase + Jupyter notebooks

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Dataset" alt="Dataset"/>

**PhreshPhish** – large-scale, high-quality phishing webpage dataset.

| Split | Samples | Benign | Phish |
|-------|---------|--------|-------|
| Train | ~498k   | ~277k  | ~222k |
| Test  | ~168k   | ~91k   | ~77k  |

- Source: [Hugging Face – phreshphish/phreshphish](https://huggingface.co/datasets/phreshphish/phreshphish)
- Contains: URL + full HTML + label + metadata
- License: CC BY 4.0 (research use)

> We use **streaming** mode so the full dataset is never downloaded to disk.

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Project+Structure" alt="Project Structure"/>

```text
phreshphish-classifier/
├── config/
│   └── config.yaml
├── notebooks/
│   ├── 01_eda_phreshphish.ipynb
│   ├── 02_preprocessing_nlp.ipynb
│   ├── 03_ml_baselines.ipynb
│   ├── 04_dl_nlp_training.ipynb
│   ├── 05_evaluation_benchmarks.ipynb
│   └── 06_model_export.ipynb
├── src/
│   ├── data/          # loader + preprocessor
│   ├── features/      # URL & HTML features
│   ├── models/        # ML, DL, ensemble
│   ├── evaluation/    # metrics
│   └── utils/
├── app/
│   ├── main.py        # Gradio entrypoint
│   ├── inference.py
│   └── ui.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── models/            # saved checkpoints
├── processed/         # cleaned parquet
├── requirements.txt
└── README.md
```

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Installation" alt="Installation"/>

```bash
# Clone the repository
git clone <your-repo-url>
cd phreshphish-classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Quick+Start" alt="Quick Start"/>

### 1. Run the notebooks (recommended order)

```bash
jupyter notebook notebooks/
```

1. `01_eda_phreshphish.ipynb` – explore the data  
2. `02_preprocessing_nlp.ipynb` – clean HTML & extract features  
3. `03_ml_baselines.ipynb` – train XGBoost  
4. `04_dl_nlp_training.ipynb` – fine-tune Transformer  
5. `05_evaluation_benchmarks.ipynb` – evaluate under realistic base rates  
6. `06_model_export.ipynb` – package models for the app  

### 2. Launch the Gradio app

```bash
python app/main.py
```

Then open `http://localhost:7860` in your browser.

### 3. CLI training / evaluation

```bash
# Train classical model
python scripts/train.py --model-type ml

# Evaluate both models
python scripts/evaluate.py --model both
```

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Models" alt="Models"/>

| Model | Type | Strengths |
|-------|------|---------|
| **XGBoost** | Classical ML | Fast, strong baseline, interpretable features |
| **DistilBERT** (or DeBERTa/GTE) | Transformer | Best Average Precision, captures context |
| **Ensemble** | Soft voting / AND | Lowest false-positive rate |

Decision thresholds are tuned for **high Precision at 90–95% Recall**.

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Evaluation+Focus" alt="Evaluation Focus"/>

Because real-world phishing base rates are very low (0.05%–5%), we report:

- **Average Precision (AP)**
- **Precision @ Recall = 0.90 / 0.95**
- **ROC-AUC**
- Performance under simulated realistic base rates

This avoids overly optimistic results common in balanced test sets.

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Configuration" alt="Configuration"/>

All major hyperparameters live in `config/config.yaml`:

- Dataset streaming options
- Preprocessing limits
- ML & DL training settings
- Low-FP evaluation thresholds
- Gradio app settings

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=License" alt="License"/>

This project is released under the **MIT License**.

```text
MIT License

Copyright (c) 2026 PhreshPhish Classifier Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> **Note**: The PhreshPhish dataset itself is licensed under **CC BY 4.0** and should only be used for anti-phishing research.

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Citation" alt="Citation"/>

If you use this project or the PhreshPhish dataset, please cite:

```bibtex
@article{dalton2025phreshphish,
  title   = {PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark},
  author  = {Thomas Dalton and Hemanth Gowda and Girish Rao and Sachin Pargi and Alireza Hadj Khodabakhshi and Joseph Rombs and Stephan Jou and Manish Marwah},
  year    = {2025},
  journal = {arXiv preprint},
  url     = {https://arxiv.org/abs/2507.10854},
  eprint  = {2507.10854}
}
```

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3B82F6&vCenter=true&width=500&lines=Acknowledgements" alt="Acknowledgements"/>

- [PhreshPhish dataset](https://huggingface.co/datasets/phreshphish/phreshphish) by OpenText / the original authors
- Hugging Face `datasets` & `transformers`
- Gradio team

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=4000&pause=1200&color=3B82F6&center=true&vCenter=true&width=600&lines=Built+for+real-world+phishing+defense;Low+FP+%E2%80%A2+High+Precision+%E2%80%A2+Production-ready" alt="Footer"/>
</p>
```
