import os
import json
import tempfile

from .settings import settings


def setup_google_credentials():
    """Escribe el JSON de la Service Account a un archivo temporal
    y apunta GOOGLE_APPLICATION_CREDENTIALS a él."""
    creds_json = settings.GOOGLE_APPLICATION_CREDENTIALS_JSON

    # Validar que sea JSON válido
    json.loads(creds_json)

    temp_dir = tempfile.gettempdir()
    creds_path = os.path.join(temp_dir, "gcp_credentials.json")

    with open(creds_path, "w") as f:
        f.write(creds_json)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return creds_path