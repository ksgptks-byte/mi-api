import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# Para pruebas locales: si no hay variable de entorno, usa una clave por defecto
API_KEY = os.getenv("API_KEY", "kskey")

@app.get("/prices")
async def get_prices(symbol: str | None = None, request: Request = None):
    # 1) Comprobamos la API Key en el header x-api-key
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2) Validamos parámetro
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    # 3) Devolvemos un precio ficticio (ejemplo)
    data = {
        "symbol": symbol,
        "price": 123.45,
        "currency": "USD",
        "as_of": __import__("datetime").datetime.utcnow().isoformat() + "Z"
    }
    return JSONResponse(content=data)
