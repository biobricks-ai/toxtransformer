# Use the official Python image from the Docker Hub
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Set a noninteractive frontend to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python 3.9, pip, and Git
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    git \
    sudo \
    libxrender1

# Symlink Python 3.9 as the default Python version
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create the vscode user and group
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# Set the vscode user as the default user
USER $USERNAME

# Install PyTorch and other dependencies
RUN pip install \
    torch torchvision torchaudio \
    pandas scikit-learn \
    biobricks dvc tqdm \
    matplotlib dask[distributed] \
    rdkit selfies==2.1.1 pyspark

# Set the working directory
WORKDIR /workspace

# Copy the requirements file and install dependencies
# COPY flask_cvae/requirements.txt requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose the port the app runs on
EXPOSE 6515

# Set the default command to bash
CMD ["bash"]
