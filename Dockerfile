FROM python:3.11-slim

# 1) Força o HOME dentro de /app
ENV HOME=/app

# 2) Define onde o streamlit vai gravar configs
#    Isso faz ~ referir a /app, logo ~/.streamlit -> /app/.streamlit  
ENV STREAMLIT_HOME=/app/.streamlit

WORKDIR /app

# 3) Instala dependências
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y

# 4) Garante a pasta de configuração do Streamlit
RUN mkdir -p /app/.streamlit

# 5) Copia o código e outras pastas
COPY . .

EXPOSE 8501

# 6) Comando de inicialização do Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

