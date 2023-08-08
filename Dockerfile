FROM python:3.9

WORKDIR /app

RUN mkdir /app/logs
RUN touch /app/logs/errors.txt
RUN touch /app/logs/logs.txt

RUN curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip

COPY requirements.txt /app
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

RUN apt-get install unzip
run unzip glove.6B.zip -d .


COPY . /app

CMD ["uvicorn", "main:app", "--workers", "1", "--host=0.0.0.0", "--port", "80"]
