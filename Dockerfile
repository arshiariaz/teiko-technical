FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python load_data.py && python analysis.py

EXPOSE 7860

CMD ["gunicorn", "app_dash:server", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
