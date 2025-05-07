import traceback

print("✅ FastAPI файл загружается...")
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Логи в консоль
    ]
)
logger = logging.getLogger("latency")

BASE_DIR   = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "model_1.joblib"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация шаблонов
try:
    templates = Jinja2Templates(directory=BASE_DIR / "templates")
except Exception as e:
    logger.critical(f"Template init error: {str(e)}")
    raise


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
    logger.info("latency %.2f ms %s", process_time, request.url.path)
    return response


class ItemFromSite(BaseModel):
    session_id: Optional[str] = None
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand: Optional[str]
    device_model: Optional[str]
    device_screen_resolution: Optional[str]
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


# ── загружаем модель
if MODEL_FILE.exists():
    try:
        bundle = joblib.load(MODEL_FILE)
        PREPROCESSOR = bundle["preprocessor"]
        MODEL = bundle["model"]
        FEATURE_ORDER = bundle["feature_order"]
        print("✅ Модель успешно загружена")
    except Exception as e:
        print("❌ Ошибка при загрузке модели:")
        traceback.print_exc()
        PREPROCESSOR = MODEL = None
        FEATURE_ORDER = []
else:
    print("⚠️ MODEL_FILE не найден:", MODEL_FILE.resolve())
    PREPROCESSOR = MODEL = None
    FEATURE_ORDER: list[str] = []

@app.get("/", response_class=HTMLResponse)
def dynamic_html(request: Request):
    try:
        return templates.TemplateResponse(
            "site_page.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Template error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict")
def predict(req: ItemFromSite):
    t0 = time.perf_counter()

    df_raw = req.to_dataframe()
    processed = PREPROCESSOR.transform(df_raw)

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "prediction": int(MODEL.predict(processed)[0]),
        "probability": float(MODEL.predict_proba(processed)[0, 1]),
        "latency_ms": round(latency_ms, 2)
    }