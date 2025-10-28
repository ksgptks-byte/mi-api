import os
import uuid
from datetime import datetime, timedelta, date
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse

app = FastAPI()

# API KEY (ya con tu valor por defecto)
API_KEY = os.getenv("API_KEY", "kskey")

def require_api_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ---------------------------
#  SALUD
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------
#  PRECIOS (actual)
# ---------------------------
@app.get("/prices")
async def get_prices(symbol: Optional[str] = None, request: Request = None):
    require_api_key(request)
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    # Precio de ejemplo
    data = {
        "symbol": symbol,
        "price": 123.45,
        "currency": "USD",
        "as_of": now_iso(),
    }
    return JSONResponse(content=data)

# ---------------------------
#  PRECIOS (histórico OHLCV)
# ---------------------------
@app.get("/prices/history")
async def get_price_history(
    request: Request,
    symbol: str = Query(..., description="Símbolo, ej. AAPL"),
    start: Optional[str] = Query(None, description="ISO 8601, opcional"),
    end: Optional[str] = Query(None, description="ISO 8601, opcional"),
    interval: str = Query("1d", regex="^(1m|5m|1h|1d|1w|1mo)$"),
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[str] = Query(None),
):
    require_api_key(request)

    # Stub simple: genera 'limit' puntos hacia atrás desde ahora (ignora start/end)
    base = datetime.utcnow()
    points = []
    for i in range(limit):
        ts = (base - timedelta(days=limit - i)).replace(microsecond=0).isoformat() + "Z"
        o = 100 + i
        h = o + 2
        l = o - 2
        c = o + 0.5
        v = 1_000_000 + i * 10_000
        points.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    return {
        "symbol": symbol,
        "interval": interval,
        "points": points,
        "next_cursor": None,  # puedes poner un string si implementas paginación real
    }

# ---------------------------
#  PRECIOS (batch de símbolos)
# ---------------------------
@app.get("/prices/batch")
async def get_batch_quotes(
    request: Request,
    symbols: str = Query(..., description="Símbolos separados por comas: AAPL,MSFT,GOOG"),
):
    require_api_key(request)

    # Parseo AAPL,MSFT,GOOG -> ["AAPL","MSFT","GOOG"]
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="symbols is required (comma-separated)")

    items = []
    for i, s in enumerate(syms):
        items.append({
            "symbol": s,
            "price": 100 + i * 2 + 0.45,
            "currency": "USD",
            "as_of": now_iso(),
        })
    return {"items": items}

# ---------------------------
#  NOTICIAS
# ---------------------------
@app.get("/news")
async def get_news(
    request: Request,
    q: Optional[str] = Query(None, description="Palabra clave"),
    sources: Optional[str] = Query(None, description="Fuentes separadas por coma"),
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    start: Optional[str] = Query(None, description="ISO 8601"),
    end: Optional[str] = Query(None, description="ISO 8601"),
):
    require_api_key(request)

    src_list = [s.strip() for s in sources.split(",")] if sources else ["ExampleWire"]
    items = []
    for i in range(limit):
        items.append({
            "title": f"Titular de ejemplo #{i+1}" + (f" sobre {q}" if q else ""),
            "url": f"https://example.com/noticia-{i+1}",
            "published_at": (datetime.utcnow() - timedelta(hours=i)).replace(microsecond=0).isoformat() + "Z",
            "source": src_list[min(i, len(src_list)-1)],
            "summary": "Resumen de ejemplo. Sustituye por el texto real cuando conectes tu feed."
        })
    return {"items": items, "next_cursor": None}

# ---------------------------
#  MACRO
# ---------------------------
@app.get("/macro/indicator")
async def get_macro_indicator(
    request: Request,
    name: str = Query(..., description="Ej. CPI, GDP, Unemployment"),
    country: Optional[str] = Query(None, description="ISO-2, ej. US, ES"),
    period: Optional[str] = Query(None, description="YYYY-MM"),
):
    require_api_key(request)

    # Stub sencillo
    return {
        "name": name,
        "country": country or "US",
        "value": 3.2,
        "period": period or datetime.utcnow().strftime("%Y-%m"),
        "unit": "%",
        "source": "DemoSource"
    }

# ---------------------------
#  RAG (búsqueda en documentos)
# ---------------------------
@app.get("/rag/search")
async def rag_search(
    request: Request,
    q: str = Query(..., description="Consulta/pregunta"),
    top_k: int = Query(5, ge=1, le=50),
):
    require_api_key(request)

    results = []
    for i in range(top_k):
        results.append({
            "id": f"doc_{i+1}",
            "text": f"Fragmento de ejemplo #{i+1} para la query: {q}. Reemplaza con resultados reales.",
            "score": round(1.0 - i * 0.05, 3),
            "source": f"s3://bucket/demo/doc{i+1}.pdf#p={i+1}",
        })
    return {"query": q, "results": results}

# ---------------------------
#  BACKTESTS
# ---------------------------
@app.post("/backtests/run")
async def run_backtest(
    request: Request,
    body: dict = Body(..., description="Ver esquema BacktestRequest en el OpenAPI"),
):
    require_api_key(request)

    # Validación mínima
    if not all(key in body for key in ("strategy", "start", "end")):
        raise HTTPException(status_code=400, detail="strategy, start, end are required")
    symbols = body.get("symbols") or ["AAPL"]
    # Stub de métricas
    performance = {
        "return_pct": 42.0,
        "sharpe": 1.23,
        "max_drawdown_pct": -12.5,
        "trades": 100
    }

    # Genera 10 puntos de curva de equity
    equity_curve = []
    try:
        # Fechas equiespaciadas
        start_dt = datetime.fromisoformat(body["start"])
        end_dt = datetime.fromisoformat(body["end"])
        total_days = max((end_dt - start_dt).days, 10)
        step = max(total_days // 10, 1)
        v = 100.0
        for i in range(10):
            d = start_dt + timedelta(days=i * step)
            v *= 1.01  # 1% por punto (ficticio)
            equity_curve.append({"date": d.date().isoformat(), "value": round(v, 2)})
    except Exception:
        # Si fallan las fechas, usa hoy hacia adelante
        d0 = date.today()
        v = 100.0
        for i in range(10):
            v *= 1.01
            equity_curve.append({"date": (d0 + timedelta(days=i)).isoformat(), "value": round(v, 2)})

    return {"performance": performance, "equity_curve": equity_curve}

# ---------------------------
#  HITL (revisión humana)
# ---------------------------
@app.post("/hitl/review")
async def submit_hitl_review(
    request: Request,
    body: dict = Body(..., description="Ver esquema HITLRequest en el OpenAPI"),
):
    require_api_key(request)

    if not all(k in body for k in ("action", "details")):
        raise HTTPException(status_code=400, detail="action and details are required")

    return {
        "status": "queued",
        "review_id": f"rvw_{uuid.uuid4().hex[:12]}"
    }




