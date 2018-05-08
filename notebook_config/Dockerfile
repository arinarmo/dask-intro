FROM python:3.5.4-slim

ENV HOME_USER jupyter
ENV HOME_PASS password

# Install system requirements
RUN apt-get update
RUN apt-get install -y build-essential gcc graphviz

# Install python requirements
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Add the jupyter user and add it to sudoers
RUN useradd -m -s /bin/bash ${HOME_USER}
RUN echo "${HOME_USER}:${HOME_PASS}"|chpasswd & \
    adduser ${HOME_USER} sudo && \
    echo ${HOME_USER}' ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Prepare the environment and copy the notebook config
USER jupyter
RUN mkdir -p /home/jupyter/workspace
WORKDIR /home/jupyter/workspace
COPY . /home/jupyter/workspace
ENV PASSWORD=password

ENTRYPOINT ["jupyter", "lab"]