FROM ubuntu:latest

WORKDIR /usr/src/app

COPY src ./
COPY requirements.txt ./

RUN mkdir "../pictures"
RUN mkdir "../processed"
RUN mkdir "../results"

RUN apt-get update \
 && apt-get install --assume-yes --no-install-recommends --quiet \
        python3 \
        python3-pip

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install --no-cache-dir -r requirements.txt

ENV SECRET_KEY my_secret_key
ENV FLASK_APP server

CMD ["python3", "run.py"]