from typing import TypedDict


class EstadoVenta(TypedDict, total=False):
    # --- Entrada (viene del request / n8n) ---
    question: str
    chat_history: str
    current_time: str
    prompts: dict
    naturaleza_producto: str   # "FISICO" | "DIGITAL"
    simbolo_moneda: str
    table_name: str

    # --- Salida ---
    respuesta: str

    # --- Control de flujo (crecerá con cada nodo nuevo) ---
    etapa: str
