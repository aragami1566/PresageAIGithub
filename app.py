import os, json, base64, asyncio, audioop, threading, time, logging
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from LLM.deepinfra import DeepInfraLLM
from twilio.rest import Client as TwilioClient

load_dotenv()
logging.basicConfig(level=logging.INFO)

# === Configuration ===
AZURE_SPEECH_KEY = os.getenv("speech_api")
AZURE_REGION = os.getenv("region")
DEEPINFRA_API_KEY = os.getenv("deepinfra_key")
DEEPINFRA_BASE_URL = os.getenv("deepinfra_base_url")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_CALLER_NUMBER")  # Numéro Twilio d'où les appels seront passés
PUBLIC_HOST = "presageaichatbot-g6ghcqg8gxh5apcs.westcentralus-01.azurewebsites.net"  # ex: "xxxx.ngrok-free.app"

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
llm_client = DeepInfraLLM(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)

app = FastAPI()

# --- Fonctions de sanitization des données sensibles ---
def sanitize_context(text: str) -> str:
    """
    Remplace les données sensibles par des placeholders.
    Par exemple, "Paul" devient "<PATIENT_NAME>" et "75 ans" devient "<PATIENT_AGE>".
    """
    if text is None:
        return ""
    sanitized = text.replace("Paul", "<PATIENT_NAME>").replace("75 ans", "<PATIENT_AGE>")
    return sanitized

def restore_sensitive_data(text: str) -> str:
    """
    Remplace les placeholders par les vraies données sensibles.
    """
    if text is None:
        return ""
    restored = text.replace("<PATIENT_NAME>", "Paul").replace("<AGE>", "75")
    return restored

# --- Gestion des sessions par appel ---
class CallSession:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.conversation = []  # liste de dicts avec "patient" et "IA"
        self.current_step_index = 0
        self.context = ""
        self.conversation_plan = [
            "Salutation et Verifier l'identité du patient nom",
            "Prendre date de RDV pour prochain suivi, avec heure",
            "Se renseigner sur quelque chose de spécifique 1 (s'il mange bien)",
            "Se renseigner sur quelque chose de spécifique 2 (s'il a dormi)",
            "Parler d'un centre d'intérêt du patient 1",
            "Parler d'un centre d'intérêt du patient 2",
            "Au revoir"
        ]
        self.summary_generated = False

    def get_current_step(self) -> str:
        if self.current_step_index < len(self.conversation_plan):
            return self.conversation_plan[self.current_step_index]
        return "Suite de conversation"

    def increment_step(self):
        if self.current_step_index < len(self.conversation_plan) - 1:
            self.current_step_index += 1

    def append_conversation(self, patient_text: str, ai_text: str):
        self.conversation.append({"patient": patient_text, "IA": ai_text})
        self.context += f"Question: {patient_text}\nRéponse: {ai_text}\n"

    def save_summary(self, summary: dict):
        filename = f"conversation_summary_{self.call_sid}.json"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2, ensure_ascii=False))
        logging.info(f"Résumé sauvegardé dans {filename}")

# Dictionnaire pour stocker les sessions en cours
sessions = {}

# --- Fonctions utilitaires STT ---
def create_speech_recognizer():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=8000, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    return recognizer, push_stream

def start_azure_recognition(recognizer, on_recognized):
    def recognizing_handler(evt):
        pass  # Logging intermédiaire si nécessaire
    def recognized_handler(evt):
        text = evt.result.text
        if text:
            on_recognized(text)
    def canceled_handler(evt):
        logging.error("Azure Speech recognition canceled: %s", evt)
    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)
    recognizer.canceled.connect(canceled_handler)
    recognizer.start_continuous_recognition()

# --- Fonctions Twilio ---
def update_call_with_twilio_tts(call_sid, text):
    if PUBLIC_HOST.startswith("http://") or PUBLIC_HOST.startswith("https://"):
        public_url = PUBLIC_HOST
    else:
        public_url = f"https://{PUBLIC_HOST}"
    twiml_response = f'''
    <Response>
        <Say language="fr-FR" voice="Polly.Lea-Neural">{text}</Say>
        <Redirect>{public_url}/incoming-call?redirected=true</Redirect>
    </Response>
    '''
    try:
        twilio_client.calls(call_sid).update(twiml=twiml_response)
        logging.info("Appel %s mis à jour pour jouer le TTS.", call_sid)
    except Exception as e:
        logging.error("Erreur lors de la mise à jour de l'appel avec Twilio TTS: %s", e)

def update_call_schedule(call_sid: str, summary: dict):
    next_appt = summary.get("next_appointment_datetime")
    if next_appt:
        try:
            if os.path.exists("call_schedule.json"):
                with open("call_schedule.json", "r", encoding="utf-8") as f:
                    schedule = json.load(f)
            else:
                schedule = {}
            schedule[call_sid] = next_appt
            with open("call_schedule.json", "w", encoding="utf-8") as f:
                json.dump(schedule, f, indent=2)
            logging.info("Call schedule updated for call %s", call_sid)
        except Exception as e:
            logging.error("Error updating call_schedule.json: %s", e)
    else:
        logging.warning("No 'next_appointment_datetime' found in summary for call %s", call_sid)

# --- Génération de résumé ---
def generate_summary_from_text(conversation_text: str) -> dict:
    raw_summary = llm_client.generate_summary_json(conversation_text)
    summary = json.loads(raw_summary) if isinstance(raw_summary, str) else raw_summary
    return summary

async def poll_call_status(call_sid: str, session: CallSession):
    max_wait_time = 1000
    poll_interval = 5
    elapsed = 0
    while elapsed < max_wait_time:
        try:
            call_details = twilio_client.calls(call_sid).fetch()
            logging.info("Statut de l'appel %s après %s s: %s", call_sid, elapsed, call_details.status)
            if call_details.status == "completed":
                if not session.summary_generated:
                    conversation_text = "\n".join(
                        [f"Patient: {entry['patient']}\nIA: {entry['IA']}" for entry in session.conversation]
                    )
                    if conversation_text:
                        logging.info("Génération du résumé à partir de la conversation:")
                        logging.info(conversation_text)
                        summary = generate_summary_from_text(conversation_text)
                        session.save_summary(summary)
                        update_call_schedule(call_sid, summary)
                        session.summary_generated = True
                break
        except Exception as e:
            logging.error("Erreur lors de la vérification du statut de l'appel %s: %s", call_sid, e)
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    else:
        logging.warning("Temps d'attente maximum atteint pour l'appel %s, résumé non généré.", call_sid)

# --- Endpoints FastAPI ---
@app.get("/")
async def root():
    return {"message": "Bienvenue sur le serveur de l'assistant médical."}

# Endpoint pour lancer un appel vers un numéro cible
@app.post("/make-call")
async def make_call(request: Request):
    """
    Lance un appel sortant vers le numéro passé dans le corps de la requête.
    Exemple de payload JSON : {"target_phone": "+33123456789"}
    """
    data = await request.json()
    target_phone = data.get("target_phone")
    if not target_phone:
        raise HTTPException(status_code=400, detail="Le paramètre 'target_phone' est requis.")
    
    # Construire l'URL publique pour le callback
    if PUBLIC_HOST.startswith("http://") or PUBLIC_HOST.startswith("https://"):
        public_url = PUBLIC_HOST
    else:
        public_url = f"https://{PUBLIC_HOST}"
    
    try:
        call = twilio_client.calls.create(
            from_=TWILIO_PHONE_NUMBER,         # Numéro Twilio d'où l'appel est lancé
            to=target_phone,                    # Numéro cible passé dans le payload
            url=f"{public_url}/incoming-call",  # URL pour le callback de l'appel
            method="GET",                       # Méthode HTTP utilisée par Twilio pour récupérer le TwiML
            send_digits="1234#"                 # DTMF à envoyer automatiquement
        )
        logging.info("Appel lancé vers %s, Call SID: %s", target_phone, call.sid)
        return {"message": "Appel lancé", "call_sid": call.sid}
    except Exception as e:
        logging.error("Erreur lors du lancement de l'appel: %s", e)
        raise HTTPException(status_code=500, detail="Erreur lors du lancement de l'appel.")

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request, background_tasks: BackgroundTasks):
    """
    Point d'entrée appelé par Twilio lors de la réponse de l'appel.
    """
    response = VoiceResponse()
    is_redirected = request.query_params.get("redirected", "false") == "true"
    if not is_redirected:
        response.say("Bonjour, je suis votre assistante médicale", language="fr-FR", voice="Polly.Lea-Neural")
    connect = Connect()
    ws_url = f"wss://{request.url.hostname}/media-stream"
    stream = Stream(url=ws_url)
    connect.append(stream)
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    recognizer, push_stream = create_speech_recognizer()
    transcript_lock = threading.Lock()
    current_transcript = ""
    last_recognized_time = time.time()
    silence_threshold = 1
    local_call_sid = None
    session = None  # instance de CallSession

    def on_recognized(text):
        nonlocal current_transcript, last_recognized_time
        with transcript_lock:
            current_transcript += " " + text
            last_recognized_time = time.time()
        logging.info("Texte reconnu: %s", text)

    threading.Thread(target=start_azure_recognition, args=(recognizer, on_recognized), daemon=True).start()

    async def silence_detector():
        nonlocal current_transcript, last_recognized_time, local_call_sid, session
        while True:
            await asyncio.sleep(1)
            with transcript_lock:
                elapsed = time.time() - last_recognized_time
                if current_transcript.strip() and elapsed > silence_threshold:
                    transcript_to_send = current_transcript.strip()
                    logging.info("Silence détecté. Texte brut: %s", transcript_to_send)
                    
                    # Sanitize l'intégralité des données envoyées au LLM
                    sanitized_context = sanitize_context(session.context) if session else ""
                    sanitized_step = sanitize_context(session.get_current_step()) if session else "Suite de conversation"
                    sanitized_question = sanitize_context(transcript_to_send)
                    
                    # Affichage du prompt complet qui sera envoyé au LLM
                    logging.info("Prompt envoyé au LLM (anonymisé) :\nContext: %s\nStep: %s\nQuestion: %s", 
                                 sanitized_context, sanitized_step, sanitized_question)
                    
                    response_text = await asyncio.to_thread(
                        llm_client.get_response, sanitized_context, sanitized_step, sanitized_question
                    )
                    # Restaurer les données sensibles dans la réponse obtenue
                    restored_response_text = restore_sensitive_data(response_text)
                    logging.info("Réponse du LLM après restauration: %s", restored_response_text)
                    
                    if session:
                        session.append_conversation(transcript_to_send, restored_response_text)
                        session.increment_step()
                        update_call_with_twilio_tts(session.call_sid, restored_response_text)
                    current_transcript = ""

    silence_task = asyncio.create_task(silence_detector())

    try:
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                logging.info("WebSocketDisconnect: la connexion est fermée.")
                break
            except Exception as e:
                logging.error("Erreur lors de la réception du WebSocket: %s", e)
                break

            try:
                message = json.loads(data)
            except Exception as e:
                logging.error("Erreur JSON: %s", e)
                continue

            event = message.get("event")
            if event == "start":
                logging.info("Flux média démarré")
                call_info = message.get("start", {})
                if call_info:
                    local_call_sid = call_info.get("callSid")
                    if local_call_sid:
                        logging.info("Call SID reçu: %s", local_call_sid)
                        session = sessions.get(local_call_sid)
                        if not session:
                            session = CallSession(local_call_sid)
                            sessions[local_call_sid] = session
                        asyncio.create_task(poll_call_status(local_call_sid, session))
            elif event == "media":
                media = message.get("media", {})
                payload = media.get("payload")
                if payload:
                    ulaw_data = base64.b64decode(payload)
                    pcm_data = audioop.ulaw2lin(ulaw_data, 2)
                    push_stream.write(pcm_data)
            elif event == "stop":
                logging.info("Flux média temporairement arrêté")
            else:
                logging.warning("Événement inconnu reçu: %s", event)
    except Exception as e:
        logging.error("Erreur dans la boucle WebSocket: %s", e)
    finally:
        try:
            push_stream.close()
        except Exception as e:
            logging.error("Erreur lors de la fermeture du push stream: %s", e)
        silence_task.cancel()
        try:
            await websocket.close()
        except Exception as e:
            logging.error("Erreur lors de la fermeture du WebSocket: %s", e)
        recognizer.stop_continuous_recognition()
        logging.info("Session STT terminée.")
