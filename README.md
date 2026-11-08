# OralVision AI

**Experimental oral-image classification with a Swin Transformer and FastAPI.**

OralVision AI is a research-oriented computer-vision prototype that serves a fine-tuned Swin Transformer through a FastAPI application. It provides a browser-based image upload experience and returns the model's predicted class, confidence score, and class scores.

> **Important — research use only:** This project is an experimental machine-learning demonstration. It is not a medical device and must not be used to diagnose oral cancer, rule out disease, make treatment decisions, or replace assessment by a qualified dental or medical professional. Model confidence is not a calibrated clinical probability.

## Features

- Swin Transformer inference using PyTorch and timm.
- FastAPI backend with interactive OpenAPI documentation.
- Browser interface for uploading oral images.
- JPG, PNG, WEBP, and BMP upload support; maximum upload size is 8 MB.
- CPU inference with automatic CUDA selection when available.
- Health endpoint reporting model readiness and inference device.
- JSON prediction response with predicted label, confidence, and per-class scores.
- Optional private-LAN mobile camera snapshot endpoint for compatible Android IP Webcam applications.

## Technology

| Area | Tools |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| Deep learning | PyTorch, torchvision, timm |
| Image handling | Pillow |
| Frontend | HTML, CSS, JavaScript |
| API documentation | OpenAPI / Swagger UI |

## How it works

1. The application loads a checkpoint from models/oral_swin_model.pth.
2. The checkpoint supplies the model architecture name, state dictionary, class mapping, image size, and normalization values.
3. Uploaded images are converted to RGB and preprocessed using the checkpoint's image size and normalization settings.
4. The model runs in evaluation/inference mode and produces class logits.
5. Softmax converts logits into relative class scores; the API returns the top class and score.

The checkpoint is expected to contain model_name, state_dict, class_to_idx, and label_map. It may also include img_size, mean, and std. The current predictor configures a two-class output layer. Exact class names and meanings are determined by the checkpoint's label_map; do not assume labels from the interface alone.

## Repository layout

~~~text
.
├── app/
│   ├── main.py              # FastAPI routes and upload validation
│   ├── predictor.py         # Swin Transformer checkpoint and inference
│   ├── templates/
│   │   └── index.html       # Web interface
│   └── static/
│       └── css/
│           └── style.css
├── models/
│   └── oral_swin_model.pth  # Required trained checkpoint (not included)
├── requirements.txt
└── README.md
~~~

The trained checkpoint is not included in this repository. Obtain it from your own training workflow and place it at the expected path before starting the application.

## Setup

### 1. Clone the repository

~~~bash
git clone https://github.com/malikahmad/-OralVision-AI-Oral-Cancer-Detection-Using-Deep-Learning-FastAPI-Swin-Transformer.git
cd -OralVision-AI-Oral-Cancer-Detection-Using-Deep-Learning-FastAPI-Swin-Transformer
~~~

### 2. Create and activate a virtual environment

**Windows PowerShell**

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

**Linux / macOS**

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

For GPU inference, install a PyTorch and torchvision build compatible with your CUDA driver using the official PyTorch installation selector. The application automatically falls back to CPU when CUDA is unavailable.

### 4. Add the model checkpoint

Create the model directory and copy your trained checkpoint into it:

~~~text
models/oral_swin_model.pth
~~~

The checkpoint must match the model architecture and metadata expected by app/predictor.py. Do not download or load checkpoints from untrusted sources.

### 5. Start the server

~~~bash
uvicorn app.main:app --reload
~~~

Open:

- Web interface: http://127.0.0.1:8000/
- Swagger API docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health status: http://127.0.0.1:8000/api/health

## API reference

### Health check

GET /api/health

Returns whether the model loaded successfully, the selected device, and any loading error.

### Image prediction

POST /api/predict

Send a multipart form request with an image file:

~~~bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -F "image=@sample.jpg"
~~~

The response includes prediction (label from the checkpoint's label map), confidence (top-class softmax score expressed as a percentage), scores (percentage scores for the model's classes), device, and a research-use notice. The score is a model output, not a clinical probability or measure of diagnostic certainty.

### Mobile camera snapshot

GET /api/camera/snapshot?host=192.168.1.25&port=8080

This optional route fetches a still image from a compatible IP Webcam application on a private IPv4 LAN address. Both devices must be on the same local network. Only the configured camera ports are accepted.

## Model evaluation and responsible reporting

Before reporting model performance:

- Keep training, validation, and final test sets separate.
- Use a patient-level split where multiple images can belong to the same person.
- Check for duplicates and data leakage between splits.
- Report class-wise precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate.
- Evaluate on an independent dataset collected under different conditions.
- Document dataset provenance, licensing, class balance, and limitations.

No accuracy or clinical-performance claim is made here because a verified independent evaluation report is not included.

## Security and privacy notes

- The API validates image type and size before inference.
- Uploaded images are processed in memory by the prediction route.
- Run the development server on trusted networks only.
- Do not expose the development server publicly without production hardening and appropriate access controls.
- The camera snapshot endpoint is intended for a private LAN; do not weaken its private-address and port restrictions.

## Current limitations

- A compatible trained checkpoint must be supplied separately.
- Predictions depend on the checkpoint's training data, label mapping, and preprocessing.
- Softmax scores may be poorly calibrated.
- The application has not been validated for clinical use.

## Author

**Ahmad Zulfiqar**  
GitHub: [@malikahmad](https://github.com/malikahmad)
