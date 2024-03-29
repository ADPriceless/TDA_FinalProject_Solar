FROM python:3.11.8-slim

WORKDIR /solar

COPY ./preprocess/transforms.py /solar/preprocess/transforms.py
COPY ./models /solar/models
COPY ./saved_models/model_20240328_074804_4 /solar/saved_models/model_20240328_074804_4
COPY ./app.py /solar/app.py

RUN pip install -r requirements.txt

# default port for streamlit is 8501
EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
