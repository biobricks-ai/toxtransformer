# docker build -t pdaa-app -f ./streamlit/Dockerfile ./
# development: docker run -p 8501:8501 -v ./:/app pdaa-app
# deploy with .aws/deploy.sh
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Install missing dependencies for Conda and Java
RUN apt-get update && apt-get install -y \
    libarchive-dev \
    openjdk-17-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy environment.yml to the container
COPY environment.yml .

# Create the Conda environment and clean up afterwards
RUN conda env create -f environment.yml && \
    conda clean -afy

# Add Conda environment to the PATH (for global access)
ENV PATH="/opt/conda/envs/pdaa/bin:$PATH"

# Switch back to default shell temporarily
SHELL ["/bin/bash", "-c"]

# Copy application code into the container
# COPY blazegraph.jar blazegraph.jnl .
# COPY ./brick/ /app/brick/
COPY ./stages/ /app/stages/

# Create streamlit directory and copy files
RUN mkdir -p /app/streamlit
COPY ./streamlit/ /app/streamlit/
# Create symlink like the source repo has
RUN ln -sf /app/streamlit/streamlit-app.py /app/streamlit-app.py

# Set execution permissions on the script
RUN chmod +x /app/streamlit/start.sh

# Re-enable Conda shell for subsequent commands
SHELL ["conda", "run", "-n", "pdaa", "/bin/bash", "-c"]

# Set the shell script as the entry point
ENTRYPOINT ["/app/streamlit/start.sh"]
