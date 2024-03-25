# Use an NVIDIA CUDA base image that includes CUDA and cuDNN
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as builder

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 without using PPA to avoid potential compatibility issues
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.9 python3.9-distutils python3.9-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Update alternatives to prioritize Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Update the symbolic link for python to point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# copy requirements file and install
COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt

# copy src code into image and chmod scripts
COPY src /opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory
WORKDIR /opt/src

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/src:${PATH}" \
    TORCH_HOME="/opt" \
    MPLCONFIGDIR="/opt"

RUN chown -R 1000:1000 /opt && \
    chmod -R 777 /opt

# Set non-root user
USER 1000

# Set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
