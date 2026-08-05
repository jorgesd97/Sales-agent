from agents.nodes.bienvenida import (
    nodo_bienvenida,
    SALUDO_PLANTILLA_DEFAULT,
    SALUDO_CORTO_DEFAULT,
)


def test_cliente_nuevo_recibe_plantilla():
    out = nodo_bienvenida({"chat_history": "", "prompts": {}})
    assert out["respuesta"] == SALUDO_PLANTILLA_DEFAULT
    assert out["etapa"] == "bienvenida"


def test_cliente_con_historial_recibe_corto():
    out = nodo_bienvenida(
        {"chat_history": "[2026-08-05 14:00] Human: hola", "prompts": {}}
    )
    assert out["respuesta"] == SALUDO_CORTO_DEFAULT
    assert out["etapa"] == "bienvenida"


def test_plantilla_de_n8n_gana_al_default():
    out = nodo_bienvenida(
        {"chat_history": "", "prompts": {"saludo_plantilla": "Hola custom PlantHero"}}
    )
    assert out["respuesta"] == "Hola custom PlantHero"


def test_historial_solo_espacios_cuenta_como_vacio():
    out = nodo_bienvenida({"chat_history": "   \n  ", "prompts": {}})
    assert out["respuesta"] == SALUDO_PLANTILLA_DEFAULT
