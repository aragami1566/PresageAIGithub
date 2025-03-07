from apscheduler.schedulers.blocking import BlockingScheduler
import pandas as pd
import requests
from datetime import datetime

def make_call(number):
    url = "https://b7c2-78-196-182-205.ngrok-free.app/make-call"
    payload = {"target_phone": number}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Appel à {number} - Statut : {response.status_code}")
        if response.status_code == 200:
            print("Réponse du serveur:", response.json())
        else:
            print("Erreur:", response.text)
    except Exception as e:
        print(f"Échec de l'appel à {number} : {str(e)}")

def start_scheduled_calls():
    # Lire le CSV avec parsing des dates ISO
    patients = pd.read_csv(
        "../call_schedule.csv",
        parse_dates=['time'],
        date_format='ISO8601'  # Format 2025-03-13T14:00:00
    )
    
    scheduler = BlockingScheduler()
    
    for _, row in patients.iterrows():
        # Extraire heure et minute depuis la datetime
        call_time = row['time'].to_pydatetime()
        
        scheduler.add_job(
            make_call,
            'cron',
            year=call_time.year,
            month=call_time.month,
            day=call_time.day,
            hour=call_time.hour,
            minute=call_time.minute,
            args=[row['number']]
        )
        print(f"Appel programmé le {call_time} au {row['number']}")

    scheduler.start()

if __name__ == "__main__":
    start_scheduled_calls()