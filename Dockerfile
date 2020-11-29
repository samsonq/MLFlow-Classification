FROM continuumio/miniconda:4.5.4

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt install -y python3

COPY requirements.txt /MLFlow-Classification/requirements.txt
RUN pip install -r /MLFlow-Classification/requirements.txt

WORKDIR /MLFlow-Classification/src
CMD python3 main.py
