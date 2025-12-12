FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command (you can override)
CMD ["python", "src/day1_clean.py", "--input", "data_raw/train.csv", "--output", "data_clean/Superstore_Cleaned.xlsx"]
