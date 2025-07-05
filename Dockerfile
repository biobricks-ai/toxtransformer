# docker build -t biobricks-ai/cvae .
# docker run -p 6515:6515 -v .:/chemsim --rm --gpus all -it --name chemsim biobricks-ai/cvae
# docker run -p 6515:6515 --rm --gpus all -it --name chemsim 010438487580.dkr.ecr.us-east-1.amazonaws.com/biobricks/chemprop-transformer
# curl "http://localhost:6515/predict?property_token=5042&inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# ----------- System Environment -----------
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_DIR=/app
ENV FLASK_APP=flask_cvae.app
ENV ROOT_URL=http://localhost:6515
ENV PORT=6515
ENV PATH="/root/.local/bin:$PATH"

# ----------- Install Python 3.11 and Tools -----------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    git \
    curl \
    libxrender1 \
    openjdk-11-jdk && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -s /usr/bin/python3.11 /usr/bin/python

# ----------- Install pipx and uv -----------
RUN pip install pipx && \
    pipx ensurepath && \
    pipx install uv

# ----------- Install Python Dependencies -----------
RUN uv pip install --system \
    flask \
    werkzeug \
    pandas \
    numpy \
    torch torchvision torchaudio \
    tqdm \
    rdkit \
    scikit-learn \
    pyyaml \
    selfies \
    faiss-cpu \
    pyspark \
    rotary_embedding_torch \
    x_transformers

# ----------- Set Up Application -----------
WORKDIR ${APP_DIR}

COPY flask_cvae/requirements.txt requirements.txt
RUN uv pip install --system -r requirements.txt

COPY flask_cvae ${APP_DIR}/flask_cvae
COPY brick/moe ${APP_DIR}/brick/moe
COPY brick/cvae.sqlite ${APP_DIR}/brick/cvae.sqlite
COPY brick/selfies_property_val_tokenizer ${APP_DIR}/brick/selfies_property_val_tokenizer
COPY cvae ${APP_DIR}/cvae

EXPOSE ${PORT}

CMD ["gunicorn", "-b", "0.0.0.0:6515", "--timeout", "480", "--graceful-timeout", "480", \
     "--workers", "1", "--keep-alive", "300", "flask_cvae.app:app"]
