# Use an TensorFlow-GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu as builder


# Install software-properties-common and python3-apt without cleaning to ensure all tools are available
RUN apt-get update && \
    apt-get install -y software-properties-common python3-apt

# Temporarily switch to the system's default Python for apt operations
RUN update-alternatives --set python /usr/bin/python3

# Add the deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils

# Set Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --config python3

# Install pip for Python 3.9
RUN apt-get install -y curl && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Clean up APT when done
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
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
