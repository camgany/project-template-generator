# Usa la imagen de Python 3.11 slim como base
FROM python:3.11-slim

# Define el argumento para la clave de OpenAI y establece la variable de entorno
ARG OPENAI_KEY
ENV OPENAI_KEY=$OPENAI_KEY

# Establece el puerto en el que la aplicación se ejecutará dentro del contenedor
ENV PORT 8000

# Copia el archivo de requisitos y realiza la instalación de dependencias
COPY requirements.txt /
RUN pip install -r requirements.txt

# Copia el directorio de código fuente a /src en el contenedor
COPY ./src /src

# Instala las bibliotecas libgl1-mesa-glx y libglib2.0-0
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Comando para ejecutar la aplicación con Uvicorn
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
