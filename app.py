import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

API_KEY = os.getenv("API_KEY", "kskey")

# --- RUTA DE SALUD ---
@app.get("/health")
def health():
    return {"ok": True}

# --- RUTA DE PRECIOS ---
@app.get("/prices")
async def get_prices(symbol: str | None = None, request: Request = None):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    data = {
        "symbol": symbol,
        "price": 123.45,
        "currency": "USD",
        "as_of": __import__("datetime").datetime.utcnow().isoformat() + "Z"
    }
    return JSONResponse(content=data)


