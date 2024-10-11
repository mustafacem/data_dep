FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  poppler-utils \
  ffmpeg \
  libgl1-mesa-glx \
  tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*


# install depends through poetry before copying the codebase
RUN pip install poetry
COPY droid-core ./droid-core
COPY pyproject.toml poetry.lock ./
COPY insight_engine/__init__.py ./insight_engine/

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# dummy entrypoint to keep the container running for development purposes
CMD ["tail", "-f", "/dev/null"]
