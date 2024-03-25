# use an TensorFlow-GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu as builder

# Install OS dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    wget \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3.11-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Update pip to the latest version
RUN python3 -m pip install --upgrade pip
RUN python3 --version

# copy requirements file and install
COPY ./requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt

# copy src code into image and chmod scripts
COPY src ./opt/src
COPY ./entry_point.sh /opt/
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/entry_point.sh
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory
WORKDIR /opt/src

# set python environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"

# set non-root user
USER 1000

# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
