FROM python:3.11

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "web_terminal_chat:app", "--host", "0.0.0.0", "--port", "7860"]