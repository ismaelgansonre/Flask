# Utiliser une image de base Python
FROM python:3.13.0b2

# Mettre à jour les paquets et installer les dépendances système
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier les fichiers de l'application et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port sur lequel l'application sera accessible
EXPOSE 8080

# Exécuter gunicorn
CMD ["gunicorn", "--bind","--device=/dev/video0:/dev/video0", "0.0.0.0:8080", "--worker-tmp-dir", "/dev/shm", "app:app"]
