# Use a base image with Java
FROM openjdk:8-jdk

# Set working directory
WORKDIR /app

# Install required packages
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    build-essential \
    curl

# Install Defects4J
RUN git clone https://github.com/rjust/defects4j.git /defects4j && \
    cd /defects4j && \
    ./init.sh

# Set environment variables for Defects4J
ENV DEFECTS4J_HOME /defects4j
ENV PATH $PATH:$DEFECTS4J_HOME/framework/bin

# Copy your validation script and other necessary files
COPY ./src /app/src
COPY ./outputs-g1 /app/outputs-g1
COPY ./datasets /app/datasets
COPY ./correct_patch /app/correct_patch

# Install Python dependencies if necessary
RUN pip3 install psutil

# Set entrypoint
ENTRYPOINT ["python3", "src/sf_val_d4j.py"]
