# Use an TensorFlow-GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu as builder

# Install Python 3.9
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --config python3



# Install pip for Python 3.9
RUN apt-get install -y curl && curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Update alternatives to use Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Verify Python version
RUN python3 --version

# copy requirements file and install
COPY ./requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt

# copy src code into image and chmod scripts
COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory
WORKDIR /opt/src

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
ENV TORCH_HOME="/opt"
ENV MPLCONFIGDIR="/opt"

# Adjust permissions
RUN chown -R 1000:1000 /opt \
    && chmod -R 777 /opt

# Set non-root user
USER 1000

# Set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
