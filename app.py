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

# Configuration
AZURE_SPEECH_KEY = os.getenv("speech_api")
AZURE_REGION = os.getenv("region")
DEEPINFRA_API_KEY = os.getenv("deepinfra_key")
DEEPINFRA_BASE_URL = os.getenv("deepinfra_base_url")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_CALLER_NUMBER")
PUBLIC_HOST = "presageaichatbot-g6ghcqg8gxh5apcs.westcentralus-01.azurewebsites.net"  # Utilise l'URL Azure automatique

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
llm_client = DeepInfraLLM(api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL)

app = FastAPI()

# Gestion des sessions d'appel
class CallSession:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.conversation = []
        self.current_step_index = 0
        self.conversation_plan = [
            "Salutation",
            "Prise de rendez-vous",
            "Questions santé",
            "Au revoir"
        ]

    def get_current_step(self) -> str:
        return self.conversation_plan[self.current_step_index] if self.current_step_index < len(self.conversation_plan) else "Suite"

    def increment_step(self):
        if self.current_step_index < len(self.conversation_plan) - 1:
            self.current_step_index += 1

    def append_conversation(self, patient_text: str, ai_text: str):
        self.conversation.append({"patient": patient_text, "IA": ai_text})

sessions = {}

# Configuration STT Azure
def create_speech_recognizer():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    push_stream = speechsdk.audio.PushAudioInputStream(
        stream_format=speechsdk.audio.AudioStreamFormat(samples_per_second=8000, bits_per_sample=16, channels=1)
    )
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    return speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config), push_stream

@app.get("/")
async def root():
    return {"message": "Serveur de chatbot vocal"}

# Gestion des appels Twilio
@app.post("/make-call")
async def make_call(request: Request):
    data = await request.json()
    target_phone = data.get("target_phone")
    
    if not target_phone:
        raise HTTPException(status_code=400, detail="Numéro cible requis")
    
    try:
        call = twilio_client.calls.create(
            from_=TWILIO_PHONE_NUMBER,
            to=target_phone,
            url=f"https://{PUBLIC_HOST}/incoming-call",
            method="GET"
        )
        return {"message": "Appel initié", "call_sid": call.sid}
    
    except Exception as e:
        logging.error("Erreur Twilio: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"wss://{PUBLIC_HOST}/media-stream")
    connect.append(stream)
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")

# WebSocket pour le flux média
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    recognizer, push_stream = create_speech_recognizer()
    session = None

    def on_recognized(text):
        if session:
            session.append_conversation(text, "Réponse IA")
            update_call_with_tts(session.call_sid, "Réponse générée")

    recognizer.recognized.connect(lambda evt: on_recognized(evt.result.text))
    recognizer.start_continuous_recognition()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("event") == "media":
                media = base64.b64decode(message["media"]["payload"])
                pcm_data = audioop.ulaw2lin(media, 2)
                push_stream.write(pcm_data)
                
    except WebSocketDisconnect:
        logging.info("Connexion fermée")
    finally:
        recognizer.stop_continuous_recognition()
        await websocket.close()

def update_call_with_tts(call_sid: str, text: str):
    try:
        twiml = f'''
        <Response>
            <Say voice="Polly.Lea-Neural" language="fr-FR">{text}</Say>
            <Redirect>https://{PUBLIC_HOST}/incoming-call</Redirect>
        </Response>'''
        twilio_client.calls(call_sid).update(twiml=twiml)
    except Exception as e:
        logging.error("Échec mise à jour appel: %s", e)