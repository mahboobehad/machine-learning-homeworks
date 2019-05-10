FROM python:3.6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "./ml_homeworks/start.py"]
COPY ./ ./ml_homeworks
