# Use an TensorFlow-GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu as builder


# Install build dependencies for Python
RUN apt-get update && \
    apt-get install -y build-essential libffi-dev libssl-dev zlib1g-dev \
    liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev libpq-dev

# Download and compile Python 3.8
RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz && \
    tar -xf Python-3.8.12.tar.xz && \
    cd Python-3.8.12 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall

# Clean up
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* Python-3.8.12.tar.xz Python-3.8.12

# Verify Python version
RUN python3.8 --version
RUN which python3.8


# Set Python 3.8 as the default python3, adjusting for its installation in /usr/local/bin
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.8

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8
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
