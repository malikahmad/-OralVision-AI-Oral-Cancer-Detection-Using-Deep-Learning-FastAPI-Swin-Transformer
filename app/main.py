from contextlib import asynccontextmanager
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from app.predictor import SwinPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "oral_swin_model.pth"
MAX_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
# Common ports used by mobile IP-camera apps. Keep this allowlist narrow.
CAMERA_PORTS = {8080, 8081, 4747, 5000}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = SwinPredictor(MODEL_PATH)
    yield


app = FastAPI(
    title="OralVision AI",
    description="Educational Swin Transformer image-classification prototype.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/health")
async def health(request: Request):
    predictor = request.app.state.predictor
    return {
        "status": "ready" if predictor.ready else "model_not_ready",
        "model_ready": predictor.ready,
        "device": str(predictor.device),
        "message": predictor.error,
    }


def decode_image(content: bytes) -> Image.Image:
    try:
        Image.open(BytesIO(content)).verify()
        return Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The supplied data is not a valid image.") from exc


async def read_upload(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WEBP, or BMP image.")
    content = await image.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Maximum image size is 8 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    return content


@app.post("/api/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    predictor = request.app.state.predictor
    if not predictor.ready:
        raise HTTPException(status_code=503, detail=predictor.error or "Model unavailable.")
    content = await read_upload(image)
    pil_image = decode_image(content)
    try:
        result = predictor.predict(pil_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **result,
        "notice": "Research use only. The model score is not a diagnosis or a calibrated disease probability.",
    }


@app.get("/api/camera/snapshot")
async def mobile_camera_snapshot(
    host: str = Query(..., description="Mobile phone's private IPv4 address"),
    port: int = Query(8080, ge=1, le=65535),
):
    """Fetch a still image from an Android IP Webcam-compatible /shot.jpg endpoint."""
    try:
        address = ip_address(host.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Enter a valid private IPv4 address, e.g. 192.168.1.25.") from exc

    if address.version != 4 or not address.is_private or address.is_loopback or address.is_link_local or address.is_multicast:
        raise HTTPException(status_code=400, detail="For safety, only a private LAN IPv4 address is allowed.")
    if port not in CAMERA_PORTS:
        raise HTTPException(status_code=400, detail=f"Use a supported camera port: {', '.join(map(str, sorted(CAMERA_PORTS)))}.")

    url = f"http://{address}:{port}/shot.jpg"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0), follow_redirects=False, trust_env=False) as client:
            upstream = await client.get(url)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the phone camera. Check the IP, port, Wi-Fi network, and that the camera app is running.") from exc

    if upstream.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Camera returned HTTP {upstream.status_code}. Check the camera app's snapshot URL.")
    content_type = upstream.headers.get("content-type", "").split(";")[0].lower()
    content = upstream.content
    if not content_type.startswith("image/") or not content or len(content) > MAX_BYTES:
        raise HTTPException(status_code=502, detail="Camera did not return a supported image under 8 MB.")
    decode_image(content)  # validate before returning data
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
