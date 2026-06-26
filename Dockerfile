FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

RUN mkdir -p /app/data /app/output

EXPOSE 8501

# Sandbox default: produce the ranked submission.
# For the dashboard: docker run -p 8501:8501 signalhire streamlit run src/app.py
CMD ["python", "src/rank.py"]
