import os
import uuid
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse

# --- RAG: embeddings (ligero) ---
from fastembed import TextEmbedding

# --- RAG: cliente Qdrant (opcional, si configuras variables) ---
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except Exception:
    QdrantClient = None  # si no está instalado o no se usa


app = FastAPI()

# ===========================
#  Config / Helpers
# ===========================
API_KEY = os.getenv("API_KEY", "kskey")

def require_api_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# yfinance: no queremos barra de progreso en logs
yf.pdr_override = False

# ---------------------------
#  RAG Setup (embedder + storage)
# ---------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # modelo rápido
_embedder = TextEmbedding(model_name=EMBED_MODEL)

# Opción A: almacenamiento en memoria (rápido para empezar)
_rag_mem_store: List[Dict[str, Any]] = []  # [{id,text,source,vector(list[float])}, ...]

# Opción B: Qdrant (persistente) si hay variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

_qdrant: Optional[QdrantClient] = None
VECTOR_SIZE: Optional[int] = None

def _ensure_embed_dim():
    global VECTOR_SIZE
    if VECTOR_SIZE is None:
        vec = list(_embedder.embed(["test"]))[0]
        VECTOR_SIZE = len(vec)
_ensure_embed_dim()

def _init_qdrant():
    global _qdrant
    if QDRANT_URL and QDRANT_API_KEY and QdrantClient is not None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Crear colección si no existe
        try:
            _qdrant.get_collection(QDRANT_COLLECTION)
        except Exception:
            _qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

_init_qdrant()


# ===========================
#  Endpoints
# ===========================

# SALUD
@app.get("/health")
def health():
    return {"ok": True}

# PRECIOS (actual)
@app.get("/prices")
async def get_prices(symbol: Optional[str] = None, request: Request = None):
    require_api_key(request)
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df is None or df.empty:
            df = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for symbol {symbol}")
        last_row = df.tail(1).iloc[0]
        price = float(last_row["Close"])
        ts = last_row.name.to_pydatetime() if hasattr(last_row.name, "to_pydatetime") else datetime.utcnow()
        return {"symbol": symbol, "price": price, "currency": "USD", "as_of": ts.replace(microsecond=0).isoformat() + "Z"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prices error: {str(e)}")

# PRECIOS (histórico)
@app.get("/prices/history")
async def get_price_history(
    request: Request,
    symbol: str = Query(...),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    interval: str = Query("1d", regex="^(1m|5m|1h|1d|1w|1mo)$"),
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[str] = Query(None),
):
    require_api_key(request)
    interval_map = {"1m":"1m","5m":"5m","1h":"60m","1d":"1d","1w":"1wk","1mo":"1mo"}
    yf_interval = interval_map[interval]
    kwargs = {"interval": yf_interval, "progress": False, "auto_adjust": True}
    if start: kwargs["start"] = start
    if end:   kwargs["end"] = end
    if not start and not end:
        kwargs["period"] = "6mo"
    try:
        df = yf.download(symbol, **kwargs)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No history for {symbol}")
        df = df.tail(limit)
        points = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else datetime.utcnow()
            points.append({
                "ts": ts.replace(microsecond=0).isoformat() + "Z",
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low":  float(row["Low"]),
                "close":float(row["Close"]),
                "volume": int(row.get("Volume", 0)),
            })
        return {"symbol": symbol, "interval": interval, "points": points, "next_cursor": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {str(e)}")

# PRECIOS (lote)
@app.get("/prices/batch")
async def get_batch_quotes(request: Request, symbols: str = Query(...)):
    require_api_key(request)
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="symbols is required (comma-separated)")
    items = []
    try:
        for s in syms:
            df = yf.download(s, period="1d", interval="1m", progress=False, auto_adjust=True)
            if df is None or df.empty:
                df = yf.download(s, period="5d", interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            last = df.tail(1).iloc[0]
            price = float(last["Close"])
            ts = last.name.to_pydatetime() if hasattr(last.name, "to_pydatetime") else datetime.utcnow()
            items.append({"symbol": s, "price": price, "currency": "USD", "as_of": ts.replace(microsecond=0).isoformat() + "Z"})
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"batch error: {str(e)}")

# NOTICIAS (NewsAPI)
@app.get("/news")
async def get_news(
    request: Request,
    q: Optional[str] = Query(None),
    sources: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    require_api_key(request)
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=501, detail="NEWS_API_KEY not configured in environment")
    params = {"pageSize": min(limit, 100), "sortBy": "publishedAt", "language": "en", "apiKey": api_key}
    if q: params["q"] = q
    if sources: params["sources"] = ",".join([s.strip().lower() for s in sources.split(",") if s.strip()])
    if start: params["from"] = start
    if end: params["to"] = end
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"news provider error: {r.text[:200]}")
        data = r.json()
        items = []
        for a in data.get("articles", [])[:limit]:
            items.append({
                "title": a.get("title"),
                "url": a.get("url"),
                "published_at": a.get("publishedAt"),
                "source": (a.get("source") or {}).get("name"),
                "summary": a.get("description"),
            })
        return {"items": items, "next_cursor": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"news error: {str(e)}")

# MACRO (FRED, USA)
@app.get("/macro/indicator")
async def get_macro_indicator(
    request: Request,
    name: str = Query(..., description="CPI, GDP, UNEMPLOYMENT"),
    country: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
):
    require_api_key(request)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise HTTPException(status_code=501, detail="FRED_API_KEY not configured in environment")
    series_map = {"CPI": "CPIAUCSL", "GDP": "GDP", "UNEMPLOYMENT": "UNRATE"}
    sid = series_map.get(name.upper())
    if (country and country.upper() != "US") or not sid:
        raise HTTPException(status_code=400, detail="Only US CPI/GDP/UNEMPLOYMENT supported in this demo")
    params = {"series_id": sid, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 1}
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=15)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"fred error: {r.text[:200]}")
        obs = r.json().get("observations", [])
        if not obs:
            raise HTTPException(status_code=404, detail="No observations")
        o = obs[0]
        val = None if o.get("value") in (None, ".") else float(o["value"])
        return {"name": name.upper(), "country": "US", "value": val, "period": o.get("date", "")[:7], "unit": "%", "source": "FRED"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"macro error: {str(e)}")

# ===== RAG =====
# Ingesta (subir/actualizar documentos) - útil para tener persistencia o memoria de trabajo
@app.post("/rag/upsert")
async def rag_upsert(request: Request, body: Dict[str, Any] = Body(...)):
    """Body: { "items": [ {"id":"doc1","text":"...","source":"..."} ] }"""
    require_api_key(request)
    items = body.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items is required")
    texts = [it["text"] for it in items]
    vectors = list(_embedder.embed(texts))
    # Opción B: Qdrant
    if _qdrant:
        points = []
        for it, vec in zip(items, vectors):
            points.append(PointStruct(id=it["id"], vector=vec, payload={"text": it["text"], "source": it.get("source","")}))
        _qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        return {"upserted": len(points), "backend": "qdrant"}
    # Opción A: memoria
    for it, vec in zip(items, vectors):
        # reemplazar si ya existe
        _rag_mem_store[:] = [d for d in _rag_mem_store if d["id"] != it["id"]]
        _rag_mem_store.append({"id": it["id"], "text": it["text"], "source": it.get("source",""), "vector": vec})
    return {"upserted": len(items), "backend": "memory"}

# Búsqueda
@app.get("/rag/search")
async def rag_search(request: Request, q: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    require_api_key(request)
    qvec = list(_embedder.embed([q]))[0]
    # Qdrant
    if _qdrant:
        res = _qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
        results = []
        for pt in res:
            results.append({
                "id": str(pt.id),
                "text": pt.payload.get("text",""),
                "score": float(pt.score),
                "source": pt.payload.get("source","")
            })
        return {"query": q, "results": results}
    # Memoria (coseno)
    if not _rag_mem_store:
        return {"query": q, "results": []}
    def cos(a, b):
        a = np.array(a); b = np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
    scored = [{"id": d["id"], "text": d["text"], "source": d["source"], "score": cos(qvec, d["vector"])} for d in _rag_mem_store]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": scored[:top_k]}

# ===== Backtests (EMA cross real simple) =====
@app.post("/backtests/run")
async def run_backtest(request: Request, body: Dict[str, Any] = Body(...)):
    require_api_key(request)
    if not all(k in body for k in ("strategy", "start", "end")):
        raise HTTPException(status_code=400, detail="strategy, start, end are required")
    strategy = body["strategy"]
    start = body["start"]
    end = body["end"]
    symbols = body.get("symbols") or ["AAPL"]
    params = body.get("params") or {"fast": 12, "slow": 26}
    if strategy.lower() != "ema_cross":
        raise HTTPException(status_code=400, detail="Only ema_cross supported in this demo")

    sym = symbols[0]  # demo: 1 símbolo
    df = yf.download(sym, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {sym} in range.")
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    if fast >= slow:
        raise HTTPException(status_code=400, detail="fast must be < slow")
    df["ema_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)  # entrar al siguiente día
    df["ret"] = df["Close"].pct_change().fillna(0)
    df["strat_ret"] = df["position"] * df["ret"]
    df["equity"] = (1 + df["strat_ret"]).cumprod() * 100.0

    # métricas
    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100.0
    daily = df["strat_ret"]
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if daily.std() > 0 else 0.0
    # max drawdown
    roll_max = df["equity"].cummax()
    drawdown = (df["equity"] / roll_max - 1.0) * 100.0
    mdd = float(drawdown.min())

    # curva
    eq = [{"date": d.strftime("%Y-%m-%d"), "value": round(v, 2)} for d, v in zip(df.index, df["equity"])]

    perf = {"return_pct": round(float(total_return), 2), "sharpe": round(float(sharpe), 3), "max_drawdown_pct": round(mdd, 2), "trades": int(df["signal"].diff().abs().sum())}
    return {"performance": perf, "equity_curve": eq}

# ===== HITL (envía a Slack) =====
@app.post("/hitl/review")
async def submit_hitl_review(request: Request, body: Dict[str, Any] = Body(...)):
    require_api_key(request)
    if not all(k in body for k in ("action", "details")):
        raise HTTPException(status_code=400, detail="action and details are required")
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        raise HTTPException(status_code=501, detail="SLACK_WEBHOOK_URL not configured in environment")
    text = f":inbox_tray: *Nueva revisión* \n• *Acción:* {body.get('action')}\n• *Detalles:* {body.get('details')}\n• *Prioridad:* {body.get('priority','normal')}\n• *Fecha:* {now_iso()}"
    try:
        r = requests.post(webhook, json={"text": text}, timeout=10)
        if r.status_code >= 300:
            raise HTTPException(status_code=502, detail=f"Slack error: {r.text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slack error: {str(e)}")
    return {"status": "queued", "review_id": f"rvw_{uuid.uuid4().hex[:12]}"}






