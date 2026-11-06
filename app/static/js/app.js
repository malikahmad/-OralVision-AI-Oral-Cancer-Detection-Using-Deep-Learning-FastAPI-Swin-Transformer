
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const analyze = document.getElementById("analyze");
const previewWrap = document.getElementById("preview-wrap");
const preview = document.getElementById("preview");
const fileInfo = document.getElementById("file-info");

const empty = document.getElementById("empty-state");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorState = document.getElementById("error-state");

let selectedFile = null;
let previewUrl = null;

function hideAllResults() {
    [empty, loading, results, errorState].forEach(
        element => element.classList.add("hidden")
    );
}

function showError(message) {
    hideAllResults();
    document.getElementById("error-message").textContent = message;
    errorState.classList.remove("hidden");
    document.getElementById("result-status").textContent = "ERROR";
}

function chooseFile(file) {
    if (!file) return;

    const validTypes = [
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    ];

    if (!validTypes.includes(file.type)) {
        showError("Choose a JPG, PNG, WEBP, or BMP image.");
        return;
    }

    if (file.size > 8 * 1024 * 1024) {
        showError("The maximum allowed image size is 8 MB.");
        return;
    }

    selectedFile = file;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    fileInfo.textContent =
        `${file.name} · ${(file.size / 1048576).toFixed(2)} MB`;

    previewWrap.classList.remove("hidden");
    dropzone.classList.add("hidden");
    analyze.disabled = false;

    hideAllResults();
    empty.classList.remove("hidden");
    document.getElementById("result-status").textContent = "WAITING";
}

function clearFile() {
    selectedFile = null;
    input.value = "";

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;

    preview.removeAttribute("src");
    previewWrap.classList.add("hidden");
    dropzone.classList.remove("hidden");
    analyze.disabled = true;

    hideAllResults();
    empty.classList.remove("hidden");
    document.getElementById("result-status").textContent = "WAITING";
}

input.addEventListener("change", () => chooseFile(input.files[0]));

dropzone.addEventListener("click", () => input.click());

dropzone.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
    }
});

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", event => {
    chooseFile(event.dataTransfer.files[0]);
});

document.getElementById("remove-image")
    .addEventListener("click", clearFile);

function showPrediction(data) {
    hideAllResults();
    results.classList.remove("hidden");

    document.getElementById("result-status").textContent = "COMPLETE";

    const prediction = data.prediction;
    const predictionElement = document.getElementById("prediction");

    predictionElement.textContent = prediction;
    predictionElement.classList.toggle(
        "cancer",
        prediction.toLowerCase() === "cancer"
    );

    document.getElementById("prediction-caption").textContent =
        prediction === "Cancer"
        ? "The model assigned this image to the Cancer category. This is not a diagnosis."
        : "The model assigned this image to the Non-Cancer category. This does not rule out disease.";

    const confidence = Math.max(
        0, Math.min(100, Number(data.confidence) || 0)
    );

    document.getElementById("confidence").textContent =
        `${confidence.toFixed(2)}%`;

    document.getElementById("confidence-bar").style.width =
        `${confidence}%`;

    document.getElementById("cancer-score").textContent =
        `${Number(data.scores?.Cancer ?? 0).toFixed(2)}%`;

    document.getElementById("normal-score").textContent =
        `${Number(data.scores?.["Non-Cancer"] ?? 0).toFixed(2)}%`;

    document.getElementById("notice").textContent =
        data.notice || "Experimental model score only; not a diagnosis.";
}

analyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideAllResults();
    loading.classList.remove("hidden");

    document.getElementById("result-status").textContent = "PROCESSING";
    document.getElementById("button-text").textContent = "Analyzing...";
    analyze.disabled = true;

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction request failed.");
        }

        showPrediction(data);

    } catch (error) {
        showError(
            error.message || "Unable to connect to the prediction API."
        );
    } finally {
        analyze.disabled = !selectedFile;
        document.getElementById("button-text").textContent =
            "Analyze image";
    }
});

async function checkModelStatus() {
    const dot = document.getElementById("health-dot");
    const label = document.getElementById("health-text");

    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        if (data.model_ready) {
            dot.classList.add("online");
            label.textContent = `Model ready · ${data.device}`;
        } else {
            dot.classList.remove("online");
            label.textContent = "Model checkpoint missing";
        }
    } catch {
        label.textContent = "API unavailable";
        dot.classList.remove("online");
    }
}

checkModelStatus();


/* ==================================================
   OralVision AI — Live Webcam Capture
   Uses the existing POST /api/predict endpoint.
   ================================================== */

(() => {
    "use strict";

    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraCanvas");
    const placeholder = document.getElementById("cameraPlaceholder");
    const capturedImage = document.getElementById("capturedImage");

    const startBtn = document.getElementById("startCameraBtn");
    const captureBtn = document.getElementById("capturePhotoBtn");
    const retakeBtn = document.getElementById("retakePhotoBtn");
    const analyzeBtn = document.getElementById("analyzePhotoBtn");
    const stopBtn = document.getElementById("stopCameraBtn");

    const status = document.getElementById("cameraStatus");
    const resultPanel = document.getElementById("cameraResult");
    const predictionText = document.getElementById("cameraPrediction");
    const confidenceText = document.getElementById("cameraConfidence");
    const scoresContainer = document.getElementById("cameraScores");

    if (!video || !canvas || !startBtn) return;

    let stream = null;
    let capturedBlob = null;
    let requestInProgress = false;

    const setStatus = (message) => {
        status.textContent = message;
    };

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }

        video.srcObject = null;
        video.hidden = true;
        stopBtn.disabled = true;
        captureBtn.disabled = true;
        startBtn.disabled = false;

        if (!capturedBlob) {
            placeholder.hidden = false;
        }

        setStatus(
            capturedBlob
                ? "Camera stopped. Your captured image is still available."
                : "Camera stopped."
        );
    }

    async function startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            setStatus(
                "Camera access is unavailable. Use HTTPS or localhost, " +
                "or upload an image instead."
            );
            return;
        }

        startBtn.disabled = true;
        setStatus("Requesting camera permission...");

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 960 }
                }
            });

            video.srcObject = stream;
            video.hidden = false;
            capturedImage.hidden = true;
            placeholder.hidden = true;

            await video.play();

            captureBtn.disabled = false;
            stopBtn.disabled = false;
            startBtn.disabled = true;
            setStatus("Camera is ready. Position the image, then capture.");
        } catch (error) {
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
                stream = null;
            }

            video.srcObject = null;
            video.hidden = true;
            placeholder.hidden = false;
            startBtn.disabled = false;

            if (error.name === "NotAllowedError") {
                setStatus(
                    "Camera permission was denied. Allow camera access " +
                    "in your browser settings and try again."
                );
            } else if (
                error.name === "NotFoundError" ||
                error.name === "DevicesNotFoundError"
            ) {
                setStatus("No compatible camera was found. Upload an image instead.");
            } else {
                setStatus("Could not start the camera. Check browser permissions.");
            }

            console.error("Camera error:", error);
        }
    }

    function capturePhoto() {
        if (!stream || !video.videoWidth || !video.videoHeight) {
            setStatus("Wait until the live camera preview is ready.");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");
        if (!context) {
            setStatus("Could not prepare the captured image.");
            return;
        }

        // Save the original camera orientation in the captured photo.
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            if (!blob) {
                setStatus("Could not capture the photo. Please try again.");
                return;
            }

            capturedBlob = blob;
            capturedImage.src = URL.createObjectURL(blob);
            capturedImage.hidden = false;
            video.hidden = true;
            placeholder.hidden = true;

            captureBtn.disabled = true;
            retakeBtn.hidden = false;
            analyzeBtn.disabled = false;

            resultPanel.hidden = true;
            setStatus("Photo captured. Review it before analysis.");
        }, "image/jpeg", 0.92);
    }

    function retakePhoto() {
        capturedBlob = null;
        capturedImage.hidden = true;
        capturedImage.removeAttribute("src");
        retakeBtn.hidden = true;
        analyzeBtn.disabled = true;
        captureBtn.disabled = !stream;
        resultPanel.hidden = true;

        if (stream) {
            video.hidden = false;
            placeholder.hidden = true;
            setStatus("Camera ready. Capture another photo.");
        } else {
            placeholder.hidden = false;
            setStatus("Start the webcam to capture another photo.");
        }
    }

    async function analyzePhoto() {
        if (!capturedBlob || requestInProgress) return;

        requestInProgress = true;
        analyzeBtn.disabled = true;
        captureBtn.disabled = true;
        retakeBtn.disabled = true;
        resultPanel.hidden = true;
        setStatus("Analyzing the captured image with Swin Transformer...");

        try {
            const formData = new FormData();
            formData.append("image", capturedBlob, "webcam-capture.jpg");

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(
                    data.detail || data.message || "Image analysis failed."
                );
            }

            // Use textContent to safely display API-provided values.
            predictionText.textContent =
                "Prediction: " + String(data.prediction ?? "Unavailable");

            confidenceText.textContent =
                data.confidence == null
                    ? "Unavailable"
                    : `${Number(data.confidence).toFixed(2)}%`;

            scoresContainer.replaceChildren();

            if (data.scores && typeof data.scores === "object") {
                for (const [label, rawScore] of Object.entries(data.scores)) {
                    const score = Number(rawScore);
                    if (!Number.isFinite(score)) continue;

                    const row = document.createElement("div");
                    row.className = "camera-score-row";

                    const heading = document.createElement("div");
                    heading.className = "camera-score-heading";

                    const name = document.createElement("span");
                    name.textContent = label;

                    const value = document.createElement("strong");
                    value.textContent = `${score.toFixed(2)}%`;

                    heading.append(name, value);

                    const track = document.createElement("div");
                    track.className = "camera-score-track";

                    const fill = document.createElement("div");
                    fill.className = "camera-score-fill";
                    fill.style.width = `${Math.max(0, Math.min(100, score))}%`;

                    track.appendChild(fill);
                    row.append(heading, track);
                    scoresContainer.appendChild(row);
                }
            }

            resultPanel.hidden = false;
            setStatus("Analysis complete. This result is for research use only.");
        } catch (error) {
            setStatus(error.message || "Unable to analyze this image.");
        } finally {
            requestInProgress = false;
            analyzeBtn.disabled = !capturedBlob;
            retakeBtn.disabled = false;
            captureBtn.disabled = !stream || Boolean(capturedBlob);
        }
    }

    startBtn.addEventListener("click", startCamera);
    captureBtn.addEventListener("click", capturePhoto);
    retakeBtn.addEventListener("click", retakePhoto);
    analyzeBtn.addEventListener("click", analyzePhoto);
    stopBtn.addEventListener("click", stopCamera);

    // Release the webcam when the page is closed or refreshed.
    window.addEventListener("pagehide", stopCamera);
})();


/* Mobile IP camera: fetch a single still from the phone's /shot.jpg endpoint. */
(() => {
  "use strict";
  const hostInput = document.getElementById("mobileCameraHost");
  const portInput = document.getElementById("mobileCameraPort");
  const captureBtn = document.getElementById("captureMobileBtn");
  const analyzeBtn = document.getElementById("analyzeMobileBtn");
  const snapshot = document.getElementById("mobileSnapshot");
  const placeholder = document.getElementById("mobileCameraPlaceholder");
  const status = document.getElementById("mobileCameraStatus");
  const result = document.getElementById("mobileCameraResult");
  const prediction = document.getElementById("mobilePrediction");
  const confidence = document.getElementById("mobileConfidence");
  const scores = document.getElementById("mobileScores");
  if (!hostInput || !captureBtn) return;

  let capturedBlob = null;
  let busy = false;

  function setStatus(message) { status.textContent = message; }

  captureBtn.addEventListener("click", async () => {
    if (busy) return;
    const host = hostInput.value.trim();
    const port = Number(portInput.value);
    if (!host) {
      setStatus("Enter the phone's IP address shown in the camera app.");
      hostInput.focus();
      return;
    }
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      setStatus("Enter a valid port number.");
      portInput.focus();
      return;
    }
    busy = true;
    captureBtn.disabled = true;
    analyzeBtn.disabled = true;
    result.hidden = true;
    setStatus("Requesting a snapshot from the phone...");
    try {
      const query = new URLSearchParams({host, port: String(port)});
      const response = await fetch(`/api/camera/snapshot?${query}`, {cache: "no-store"});
      const dataType = response.headers.get("content-type") || "";
      if (!response.ok) {
        const data = dataType.includes("application/json") ? await response.json() : {};
        throw new Error(data.detail || "Could not capture a snapshot from the phone.");
      }
      capturedBlob = await response.blob();
      if (!capturedBlob.type.startsWith("image/")) throw new Error("The phone did not return an image.");
      snapshot.src = URL.createObjectURL(capturedBlob);
      snapshot.hidden = false;
      placeholder.hidden = true;
      analyzeBtn.disabled = false;
      setStatus("Snapshot received. Review the photo, then select Analyze captured image.");
    } catch (error) {
      capturedBlob = null;
      setStatus(error.message || "Could not connect to the phone camera.");
    } finally {
      busy = false;
      captureBtn.disabled = false;
    }
  });

  analyzeBtn.addEventListener("click", async () => {
    if (!capturedBlob || busy) return;
    busy = true;
    analyzeBtn.disabled = true;
    captureBtn.disabled = true;
    result.hidden = true;
    setStatus("Analyzing the mobile snapshot...");
    try {
      const form = new FormData();
      form.append("image", capturedBlob, "mobile-camera.jpg");
      const response = await fetch("/api/predict", {method: "POST", body: form});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Image analysis failed.");
      prediction.textContent = `Prediction: ${String(data.prediction ?? "Unavailable")}`;
      confidence.textContent = data.confidence == null ? "Unavailable" : `${Number(data.confidence).toFixed(2)}%`;
      scores.replaceChildren();
      for (const [label, raw] of Object.entries(data.scores || {})) {
        const value = Number(raw);
        if (!Number.isFinite(value)) continue;
        const row = document.createElement("div");
        row.className = "camera-score-row";
        const heading = document.createElement("div");
        heading.className = "camera-score-heading";
        const name = document.createElement("span");
        name.textContent = label;
        const pct = document.createElement("strong");
        pct.textContent = `${value.toFixed(2)}%`;
        heading.append(name, pct);
        const track = document.createElement("div");
        track.className = "camera-score-track";
        const fill = document.createElement("div");
        fill.className = "camera-score-fill";
        fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
        track.append(fill);
        row.append(heading, track);
        scores.append(row);
      }
      result.hidden = false;
      setStatus("Analysis complete. This result is for research use only.");
    } catch (error) {
      setStatus(error.message || "Unable to analyze the mobile snapshot.");
    } finally {
      busy = false;
      captureBtn.disabled = false;
      analyzeBtn.disabled = !capturedBlob;
    }
  });

  window.addEventListener("pagehide", () => {
    if (snapshot.src.startsWith("blob:")) URL.revokeObjectURL(snapshot.src);
  });
})();
