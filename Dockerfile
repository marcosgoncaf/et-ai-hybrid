FROM python:3.11-slim

ENV STREAMLIT_HOME=/app/.streamlit
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential && apt-get autoremove -y

RUN mkdir -p /app/.streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
