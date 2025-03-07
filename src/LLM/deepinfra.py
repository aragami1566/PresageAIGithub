import time
import json
from openai import OpenAI
import datetime
import locale

class DeepInfraLLM:
    def __init__(self, api_key, base_url, model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.system_prompt_template = """
Langue: francaise
Vous êtes un assistant médical amical, empathique et professionnel. Vous effectuez des appels téléphoniques de suivi auprès de personnes âgées. Vous parlez un excellent français sans fautes d’orthographe, en utilisant le vouvoiement. 
- Réponses courtes et concises
Tu t'appelles Catherine.
Le nom du patient est <PATIENT_NAME>, âgé de <AGE> ans. 
Votre style doit être naturel et fluide, comme lors d'une vraie conversation téléphonique : 
- Commencez par une brève salutation, puis poursuivez directement avec la question ou l'information pertinente.
- Ne pas répéter "Bonjour" à chaque réponse
- Utilisez des transitions naturelles entre les sujets.
- Posez des questions courtes et agréables
- Ne pas répéter les réponses déjà données

Suivez précisément le plan de conversation et adaptez-vous à la réponse de l'utilisateur. Parfois, la réponse peut être erronée (vu que c'est par téléphone), il faut s'adapter.

Historique de la conversation : {context}
Sujet à aborder : {step}
Ce que dit l'utilisateur : {question}
"""

    def get_response(self, context, step, question):
        system_message = {
            "role": "system",
            "content": self.system_prompt_template.format(context=context, step=step, question=question)
        }
        user_message = {"role": "user", "content": question}
    
        start_time = time.time()
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            stream=False,
            temperature=self.temperature,
        )
        end_time = time.time()
        print(f"Réponse IA ({end_time - start_time:.2f}s):")
        response = chat_completion.choices[0].message.content
        print(response)
        return response

    def generate_summary_json(self, conversation_history: str) -> dict:
        # Tenter de définir la locale en français pour obtenir le jour en français
        try:
            locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
        except Exception as e:
            print("Erreur lors du réglage de la locale:", e)
        
        now_datetime = datetime.datetime.now()
        now = now_datetime.strftime("%Y-%m-%dT%H:%M:%S")
        day_of_week = now_datetime.strftime("%A").lower()  # Exemple : "lundi", "mardi", etc.
        
        summary_prompt = f"""
    Veuillez analyser l'historique de conversation ci-dessous, qui correspond à un suivi téléphonique d'un patient âgé nommé Paul.
    Aujourd'hui, la date et l'heure actuelles sont {now} et nous sommes {day_of_week}.
    À partir de ces échanges, créez un résumé structuré au format JSON contenant les informations suivantes :

    - "patient_name": le nom du patient.
    - "age": l'âge du patient.
    - "conditions": un résumé des conditions ou remarques importantes évoquées (ex. alimentation, sommeil, centres d'intérêt, etc.).
    - "next_appointment_datetime": la prochaine date et heure de rendez-vous au format ISO 8601 (YYYY-MM-DDTHH:MM:SS) si mentionnée, sinon "None".
    - "conversation_summary": un résumé global de la conversation, en indiquant les points clés évoqués.
    - "additional_notes": tout autre élément important qui ressort de la conversation.

    Voici l'historique de la conversation :
    {conversation_history}

    Veuillez produire uniquement le JSON sans explications supplémentaires.
        """.strip()
        
        system_message = {
            "role": "system",
            "content": "Vous êtes un assistant médical expérimenté, chargé d'extraire les informations clés d'une conversation téléphonique de suivi."
        }
        user_message = {"role": "user", "content": summary_prompt}

        start_time = time.time()
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            stream=False,
            temperature=self.temperature,
        )
        end_time = time.time()
        print(f"Temps d'appel LLM (summary): {end_time - start_time:.2f} sec")

        response_text = chat_completion.choices[0].message.content.strip()
        try:
            summary_json = json.loads(response_text)
        except Exception as e:
            print("Erreur lors de la conversion du texte en JSON:", e)
            print("Réponse brute du LLM :", response_text)
            summary_json = {}

        return summary_json