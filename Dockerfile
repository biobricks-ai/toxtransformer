# docker build -t biobricks-ai/cvae .
# docker run -p 6515:6515 -v .:/chemsim --rm --gpus all -it --name chemsim biobricks-ai/cvae
# docker run -p 6515:6515 --rm --gpus all -it --name chemsim 010438487580.dkr.ecr.us-east-1.amazonaws.com/biobricks/chemprop-transformer
# curl "http://localhost:6515/predict?property_token=5042&inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Set a noninteractive frontend to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python 3.9, pip, and Git
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    git

# Symlink Python 3.9 as the default Python version
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch and other dependencies
RUN pip install \
    torch torchvision torchaudio \
    pandas scikit-learn pyarrow \
    biobricks dvc tqdm \
    matplotlib dask[distributed] \
    rdkit

# RDKIT needs libxrender1
RUN apt-get install -y libxrender1

# Install the requirements
COPY flask_cvae/requirements.txt requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the app
RUN mkdir -p app
COPY flask_cvae app/flask_cvae

# Command to run the application
ENV FLASK_APP=flask_cvae.app
ENV ROOT_URL=http://localhost:6515

# Data directories (brick/, cache/) are mounted from persistent disk at runtime
# via docker run -v /mnt/data/brick:/app/brick -v /mnt/data/cache:/app/cache
# This avoids copying 422GB+ of data into the image
RUN mkdir -p app/brick app/cache

# Expose the port the app runs on
EXPOSE 6515

# add the cvae module
COPY cvae app/cvae

# Start the container with a bash shell
WORKDIR /app
CMD ["gunicorn", "-b", "0.0.0.0:6515", "--timeout", "480", "--graceful-timeout", "480", \
    "--workers", "1", "--threads", "8", "--worker-class", "gthread", "--keep-alive", "300", "flask_cvae.app:app"]
# gunicorn -b 0.0.0.0:6515 --timeout 480 --graceful-timeout 480 --workers 1 flask_cvae.app:app