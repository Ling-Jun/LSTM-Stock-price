# Dockerfile - this is a comment. Delete me if you want.

FROM python:3.7.6

# This Dockerfile copies our current folder . , into our container folder /app
COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["Flask_webapp/app.py"]
