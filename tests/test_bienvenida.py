from agents.nodes.bienvenida import nodo_bienvenida, SALUDO_PLANTILLA_DEFAULT


def test_cliente_nuevo_devuelve_prefijo_de_saludo():
    out = nodo_bienvenida({"chat_history": "", "prompts": {}})
    assert out["saludo_prefijo"] == SALUDO_PLANTILLA_DEFAULT
    assert out["etapa"] == "consultivo"


def test_cliente_con_historial_no_saluda():
    out = nodo_bienvenida(
        {"chat_history": "[2026-08-05 14:00] Human: hola", "prompts": {}}
    )
    assert out["saludo_prefijo"] == ""


def test_plantilla_de_n8n_gana_al_default():
    out = nodo_bienvenida(
        {"chat_history": "", "prompts": {"saludo_plantilla": "¡Bienvenido a Plant Hero! 🌱"}}
    )
    assert out["saludo_prefijo"] == "¡Bienvenido a Plant Hero! 🌱"


def test_historial_solo_espacios_cuenta_como_vacio():
    out = nodo_bienvenida({"chat_history": "   \n  ", "prompts": {}})
    assert out["saludo_prefijo"] == SALUDO_PLANTILLA_DEFAULT
