# app/Dockerfile

FROM python:3.9-slim

WORKDIR .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/equinor/neqsimhome.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

# Set Java options to allow native access for JPype (required for Java 16+)
ENV JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED"

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
