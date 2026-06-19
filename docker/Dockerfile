FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY flashseg/ flashseg/

RUN pip install --no-cache-dir -e ".[all]"

COPY . .

ENTRYPOINT ["flashseg"]
CMD ["--help"]
