FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY saved_models/01.rf_model.pkl saved_models/01.rf_model.pkl

COPY main.py main.py

CMD ["python", "main.py"]
EXPOSE 5000