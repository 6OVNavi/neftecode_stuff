FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY daimler_mixtures_train.csv daimler_mixtures_test.csv daimler_component_properties.csv ./
COPY inference.ipynb ./

CMD ["python", "-m", "src.infer", "--data_dir", ".", "--artifacts", "artifacts", "--out", "predictions.csv"]
