import time
import json
from openai import OpenAI

class DeepInfraLLM:
    def __init__(self, api_key, base_url, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.1):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.system_prompt_template = """
Langue: francaise
Vous êtes un assistant médical amical, empathique et professionnel. Vous effectuez des appels téléphoniques de suivi auprès de personnes âgées. Vous parlez un excellent français sans fautes d’orthographe, en utilisant le vouvoiement. 
- Réponses courtes et concises

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
        # Enable streaming by setting stream=True
        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            stream=True,
            temperature=self.temperature,
        )
        end_time = time.time()
        print(f"Réponse IA ({end_time - start_time:.2f}s):")
        response = ""
        # Accumulate the streaming chunks
        for chunk in stream_response:
            # Each chunk's structure: chunk.choices[0].message.content
            response += chunk.choices[0].message.content
        return response

    def generate_summary_json(self, conversation_history: str) -> dict:
        summary_prompt = f"""
Veuillez analyser l'historique de conversation ci-dessous, qui correspond à un suivi téléphonique d'un patient âgé nommé Paul.
À partir de ces échanges, créez un résumé structuré au format JSON contenant les informations suivantes :

- "patient_name": le nom du patient.
- "age": l'âge du patient.
- "conditions": un résumé des conditions ou remarques importantes évoquées (ex. alimentation, sommeil, centres d'intérêt, etc.).
- "next_appointment_date": la prochaine date de rendez-vous, si mentionnée, sinon "None".
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
        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            stream=True,
            temperature=self.temperature,
        )
        end_time = time.time()
        print(f"Temps d'appel LLM (summary): {end_time - start_time:.2f} sec")
    
        response_text = ""
        for chunk in stream_response:
            response_text += chunk.choices[0].message.content.strip()
    
        try:
            summary_json = json.loads(response_text)
        except Exception as e:
            print("Erreur lors de la conversion du texte en JSON:", e)
            print("Réponse brute du LLM :", response_text)
            summary_json = {}
    
        return summary_json

    def generate_conversation_plan(self, patient_info: dict) -> list:
        patient_info_str = json.dumps(patient_info, ensure_ascii=False, indent=2)
        prompt = f"""
Veuillez utiliser les informations suivantes du patient pour créer un plan de conversation détaillé pour un appel de suivi téléphonique.
Le plan doit être au format JSON et contenir une clé "steps" qui est une liste d'étapes de conversation. Les étapes doivent inclure :
- Une salutation en mentionnant le nom et l'âge du patient.
- Des questions sur les conditions du patient.
- Une discussion sur les centres d'intérêt.
- Une proposition pour fixer la prochaine date de rendez-vous.
- La clôture de l'appel.

Informations du patient :
{patient_info_str}

Veuillez fournir uniquement le JSON sans explications supplémentaires.
        """.strip()
        
        system_message = {
            "role": "system",
            "content": "Vous êtes un assistant médical expérimenté spécialisé dans la conduite d'appels de suivi."
        }
        user_message = {"role": "user", "content": prompt}
    
        start_time = time.time()
        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            stream=True,
            temperature=self.temperature,
        )
        end_time = time.time()
        print(f"Temps d'appel LLM (conversation plan): {end_time - start_time:.2f} sec")
    
        response_text = ""
        for chunk in stream_response:
            response_text += chunk.choices[0].message.content.strip()
    
        try:
            plan_json = json.loads(response_text)
            steps = plan_json.get("steps", [])
        except Exception as e:
            print("Erreur lors de la conversion du texte en JSON pour le plan:", e)
            print("Réponse brute du LLM :", response_text)
            steps = []
    
        return steps
