# Base image
FROM python:3.8-slim

ARG INPUT_DIR
ARG OUTPUT_DIR

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY Vessel_Segmentation-multi-task Vessel_Segmentation-multi-task
COPY ${INPUT_DIR}/* Vessel_Segmentation/data


WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir

ENV NVIDIA_VISIBLE_DEVICES all
ENV N