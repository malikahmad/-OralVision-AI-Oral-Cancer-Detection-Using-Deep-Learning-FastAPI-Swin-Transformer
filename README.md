# OralVision AI — Swin Transformer Image Classification Prototype

OralVision AI is an educational image-classification prototype built with FastAPI and a Swin Transformer-based predictor. It accepts an oral-image upload and returns the model's predicted class information when the trained checkpoint is available.

> **Medical safety notice:** This project is for software engineering and research demonstration only. It is not a medical device and must not be used to diagnose oral cancer, screen patients, or guide treatment. Any concerning lesion should be assessed by a qualified dental or medical professional.

## Features

- FastAPI application with a browser-based interface.
- Swin Transformer inference through a dedicated predictor module.
- Image upload validation for JPG, PNG, WEBP, and BMP.
- Upload size limit of 8 MB.
- Model health endpoint reporting readiness and inference device.
- Prediction endpoint with a research-use notice.
- Optional mobile IP Webcam-compatible snapshot endpoint restricted to private LAN IPv4 addresses and an explicit port allowlist.

## Technology

- Python
- FastAPI and Uvicorn
- PyTorch, torchvision, and timm
- Pillow
- Jinja2 templates
- HTTPX

## Repository layout

```text
.
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── static/
│   └── templates/
├── models/
│   └── oral_swin_model.pth
├── requirements.txt
└── README.md
```

## Setup

Use a Python environment compatible with your installed PyTorch build.

```bash
git clone https://github.com/malikahmad/-OralVision-AI-Oral-Cancer-Detection-Using-Deep-Learning-FastAPI-Swin-Transformer.git
cd -OralVision-AI-Oral-Cancer-Detection-Using-Deep-Learning-FastAPI-Swin-Transformer

python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your operating system and hardware if the default dependency installation is not suitable. Place the trained checkpoint at:

```text
models/oral_swin_model.pth
```

The model checkpoint must match the architecture and preprocessing expected by `app/predictor.py`.

## Run

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/api/health` | Model readiness and device status |
| POST | `/api/predict` | Submit an image for model inference |
| GET | `/api/camera/snapshot` | Fetch a still image from an allowed private-LAN camera endpoint |

The prediction endpoint expects an image upload using the `image` form field. See the FastAPI interactive API documentation at `/docs` while the application is running.

## Model evaluation

Add results from a reproducible, held-out evaluation before publishing performance claims. Report class-wise precision, recall, F1-score, confusion matrix, dataset source, split method, and test-set size. Avoid reporting training or validation performance as independent test performance.

No accuracy or clinical-performance claim is made in this README.

## Security and limitations

- Keep the model checkpoint and any sensitive data out of public commits unless licensing and privacy have been checked.
- The camera snapshot feature is restricted to private IPv4 addresses and a defined port allowlist; do not broaden this without reviewing SSRF protections.
- The model may fail on images that differ from its training distribution.
- Confidence scores are not calibrated disease probabilities.
- This prototype has not been established as clinically validated.

## Author

**Ahmad Zulfiqar** — [GitHub](https://github.com/malikahmad) · [LinkedIn](https://www.linkedin.com/in/malik-ahmad-06abb31a1/)
