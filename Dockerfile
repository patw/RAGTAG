FROM ubuntu:23.10
ENV CONTAINER_SHELL=bash
ENV CONTAINER=

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# basic app installs
# two steps because i kept getting cache errors without it
RUN apt clean
RUN apt update
RUN apt install -y \
        python3.11  \
        wget \ 
        git \
        openssl \
        libssl-dev \
        pkg-config \
        build-essential \
        python3-pip \
        dos2unix

# links
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 -f
RUN ln -s /usr/bin/python3.11 /usr/bin/python -f

# install pip
#RUN wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py
#RUN python3 get-pip.py
RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED

RUN mkdir /opt/ragtag
COPY ./dolphin-2.1-mistral-7b.Q5_K_S.gguf /opt/ragtag/dolphin-2.1-mistral-7b.Q5_K_S.gguf
COPY ./app.py /opt/ragtag/app.py
COPY ./requirements.txt /opt/ragtag/requirements.txt
COPY ./templates /opt/ragtag/templates
COPY ./static /opt/ragtag/static
COPY ./preloadpackages.py /opt/ragtag/preloadpackages.py
COPY ./runner.sh /opt/ragtag/runner.sh
RUN chmod +x /opt/ragtag/runner.sh
RUN dos2unix /opt/ragtag/runner.sh

# install pip required packages
RUN python3 -m pip install -r /opt/ragtag/requirements.txt --break-system-packages

RUN cd /opt/ragtag
WORKDIR /opt/ragtag
RUN python3 preloadpackages.py
# flask run -p 80 --host 0.0.0.0
ENTRYPOINT ["/bin/bash", "/opt/ragtag/runner.sh"]
#CMD ["/bin/bash"]

EXPOSE 80
