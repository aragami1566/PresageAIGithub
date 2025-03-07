# Utiliser une image Python officielle (ici version 3.9-slim)
FROM python:3.10-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'application dans le container
COPY app.py .
COPY LLM ./LLM

# Exposer le port 80 (ou un autre port selon vos besoins)
EXPOSE 80

# Lancer l'application avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
