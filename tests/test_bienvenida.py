import asyncio

from agents.messages import PeticionCliente, PeticionConSaludo
from agents.nodes.bienvenida import SALUDO_PLANTILLA_DEFAULT, BienvenidaExecutor


class FakeContext:
    """Doble de WorkflowContext: solo captura lo que se envía.

    En LangGraph el nodo devolvía un dict y se comprobaba el retorno; acá el
    executor no devuelve nada, así que hay que capturar el `send_message`.
    """

    def __init__(self):
        self.enviados: list[PeticionConSaludo] = []

    async def send_message(self, message):
        self.enviados.append(message)


def ejecutar(chat_history: str = "", prompts: dict = None) -> PeticionConSaludo:
    ctx = FakeContext()
    asyncio.run(
        BienvenidaExecutor().run(
            PeticionCliente(
                question="¿tienen perlita?",
                chat_history=chat_history,
                prompts=prompts or {},
            ),
            ctx,
        )
    )
    assert len(ctx.enviados) == 1
    return ctx.enviados[0]


def test_cliente_nuevo_devuelve_prefijo_de_saludo():
    assert ejecutar(chat_history="").saludo_prefijo == SALUDO_PLANTILLA_DEFAULT


def test_cliente_con_historial_no_saluda():
    salida = ejecutar(chat_history="[2026-08-05 14:00] Human: hola")
    assert salida.saludo_prefijo == ""


def test_plantilla_de_n8n_gana_al_default():
    salida = ejecutar(
        chat_history="",
        prompts={"saludo_plantilla": "¡Bienvenido a Plant Hero! 🌱"},
    )
    assert salida.saludo_prefijo == "¡Bienvenido a Plant Hero! 🌱"


def test_historial_solo_espacios_cuenta_como_vacio():
    assert ejecutar(chat_history="   \n  ").saludo_prefijo == SALUDO_PLANTILLA_DEFAULT


def test_la_peticion_original_viaja_intacta():
    salida = ejecutar(chat_history="")
    assert salida.peticion.question == "¿tienen perlita?"
