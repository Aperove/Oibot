FROM python:3.11

# 工作目录为/app
WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
