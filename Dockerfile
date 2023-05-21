FROM circleci/python:3.10

USER root

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install default-jdk -y
RUN apt-get install curl pandoc graphviz-dev coinor-cbc coinor-libcbc-dev -y

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

WORKDIR /app
ENV ANALYTICS_MODULE_PATH=/app/analytics

COPY analytics/src/requirements.txt /analytics_requirements.txt
RUN pip install -r /analytics_requirements.txt

COPY requirements.txt /main_requirements.txt
RUN pip install -r /main_requirements.txt

RUN pip freeze

COPY . .

EXPOSE 8001

CMD ["/bin/bash", "-c", "uvicorn main:app --host '0.0.0.0' --port 8001"]