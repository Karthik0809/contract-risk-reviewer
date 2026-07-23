FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Basic system dependencies often needed by scientific/python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Streamlit runtime config (Contract Risk Reviewer is the primary app).
ENV PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

# Bind to $PORT so the image works on Hugging Face Spaces, Cloud Run, etc.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
