import httpx
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class PromotionsService:
    """
    Servicio que lee las promociones activas desde la tabla `promociones`
    de Supabase vía la API REST de PostgREST.

    A diferencia del AzureSearchRetriever (que hace búsqueda híbrida sobre la KB),
    este servicio hace un query estructurado directo a una tabla. Se usa para
    inyectar TODAS las promociones activas en el prompt del agente en cada turno,
    ordenadas por prioridad descendente.
    """

    def __init__(self):
        self.url = f"{settings.SUPABASE_URL}/rest/v1/promociones"
        self.headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
        }

    async def get_active_promotions(self) -> list[dict]:
        params = {
            "activa": "eq.true",
            "select": "nombre,descripcion,condicion_aplicacion,mensaje_gancho,cuando_usar,prioridad",
            "order": "prioridad.desc",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.url,
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                promos = response.json()

            logger.info(f"[promotions] {len(promos)} promociones activas cargadas")
            return promos

        except Exception as e:
            logger.error(f"[promotions] error al leer promociones: {e}")
            return []

    def format_promotions(self, promos: list[dict]) -> str:
        if not promos:
            return "No hay promociones activas en este momento."

        lines = []
        for i, p in enumerate(promos, 1):
            lines.append(
                f"{i}. **{p['nombre']}**\n"
                f"   - Descripción: {p['descripcion']}\n"
                f"   - Aplica cuando: {p['condicion_aplicacion']}\n"
                f"   - Frase sugerida: \"{p['mensaje_gancho']}\"\n"
                f"   - Úsala cuando: {p['cuando_usar']}"
            )
        return "\n\n".join(lines)
