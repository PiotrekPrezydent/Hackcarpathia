import socket
import os
import json
from heart_model import HeartModel  # Import klasy HeartModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/heartd_model.pkl")  # Ścieżka do modelu

def start_server():
    # Tworzenie socketu
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 8080))  # Ustaw nasłuchiwanie na wszystkich interfejsach
    server_socket.listen(5)  # Maksymalna liczba oczekujących połączeń
    print("Serwer działa na 0.0.0.0:8080")

    # Model domyślny
    model = None

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Połączono z {client_address}\n")

        try:
            # Odbieranie danych od klienta
            request_data = client_socket.recv(1024).decode('utf-8')
            print("Otrzymano dane:", request_data, "\n")

            if not request_data:
                client_socket.close()
                continue

            # Wyodrębnienie JSON z żądania HTTP
            try:
                headers, body = request_data.split("\r\n\r\n", 1)  # Oddzielenie nagłówków od treści
                input_data = json.loads(body.strip())  # Usuwanie zbędnych białych znaków
            except Exception as e:
                print("Błąd parsowania JSON:", e)
                response = json.dumps({"error": "Nieprawidłowy format danych JSON"})
                send_response(client_socket, response)
                continue

            # Weryfikacja obecności pola "model"
            if "model" not in input_data:
                print("Brak pola 'model' w danych wejściowych.")
                response = json.dumps({"error": "Pole 'model' jest wymagane."})
                send_response(client_socket, response)
                continue

            # Pobranie modelu na podstawie pola "model"
            model_name = input_data["model"]
            if model_name == "heart_model":
                if model is None:
                    model = HeartModel()
                    if os.path.exists(MODEL_PATH):
                        print("Wczytywanie modelu...")
                        model.AiModel()
                    else:
                        print("Model nie istnieje - zostanie utworzony po pierwszym żądaniu.")
            else:
                print(f"Nieobsługiwany model: {model_name}")
                response = json.dumps({"error": f"Nieobsługiwany model: {model_name}"})
                send_response(client_socket, response)
                continue

            # Usunięcie modelu na polecenie "r"
            if isinstance(input_data.get("data"), str) and input_data["data"].strip().lower() == "r":
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    model = None
                    print("Model został usunięty.")
                    response = json.dumps({"message": "Model został usunięty."})
                else:
                    print("Brak modelu do usunięcia.")
                    response = json.dumps({"message": "Brak modelu do usunięcia."})
                send_response(client_socket, response)
                continue

            # Predykcja
            prediction = model.predict(input_data.get("data"))
            decision = "Choroba serca" if prediction == 1 else "Brak choroby serca"
            response = json.dumps({"prediction": decision})
            send_response(client_socket, response)

        except Exception as e:
            print("Wystąpił błąd:", e)
            error_response = json.dumps({"error": str(e)})
            send_response(client_socket, error_response)
        finally:
            client_socket.close()

def send_response(client_socket, response_body):
    """Wysyłanie odpowiedzi HTTP do klienta."""
    response = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(response_body.encode('utf-8'))}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{response_body}"
    )
    client_socket.sendall(response.encode('utf-8'))

if __name__ == "__main__":
    start_server()
