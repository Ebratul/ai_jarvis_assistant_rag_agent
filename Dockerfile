FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nono_rag ./nono_rag
ENV NONO_DATA_DIR=/data
VOLUME /data
EXPOSE 8100
CMD ["python", "-m", "nono_rag"]
