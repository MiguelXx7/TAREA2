import os
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configuración de la API Key desde variable de entorno
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError(
        "Define la variable de entorno DEEPSEEK_API_KEY con tu API Key de DeepSeek"
    )
ENDPOINT = "https://api.deepseek.com/v1/chat/completions"


def get_session(api_key: str) -> requests.Session:
    """
    Crea y configura una sesión HTTP con retries y cabeceras base.
    """
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    })
    # Configuración de retries para errores transitorios
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def enviar_prompt(session: requests.Session, prompt: str) -> requests.Response:
    """
    Envía el prompt a la API de DeepSeek y retorna el objeto Response.
    """
    system_msg = {
        "role": "system",
        "content": (
            "Eres un asistente experto en telecomunicaciones que responde con ejemplos claros "
            "y cálculos sencillos."
        )
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [system_msg, {"role": "user", "content": prompt}]
    }
    return session.post(ENDPOINT, json=payload, timeout=5)


def chat():
    """
    Bucle principal de conversación por consola.
    """
    session = get_session(API_KEY)
    print("¡Hola! Soy tu chatbot personalizado. Escribe 'salir' para terminar.")
    while True:
        prompt = input("Tú: ")
        if prompt.strip().lower() == "salir":
            print("Chat finalizado. ¡Hasta luego!")
            break

        try:
            response = enviar_prompt(session, prompt)
            data = response.json()
        except json.JSONDecodeError:
            print("Respuesta no válida de la API:", response.text)
            continue
        except requests.RequestException as e:
            print("Error al conectar con la API:", e)
            continue

        if response.ok:
            try:
                reply = data["choices"][0]["message"]["content"]
                print(f"Bot: {reply}")
            except (KeyError, IndexError):
                print("Formato inesperado en la respuesta:", data)
        else:
            error_info = data.get("error", {})
            print(f"Error {response.status_code}: {error_info}")


if __name__ == "__main__":
    chat()
