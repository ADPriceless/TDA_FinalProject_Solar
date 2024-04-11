FROM python:3.11.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}

# -- Copy Files --
# Copy across files for app
COPY ./preprocess/transforms.py ./preprocess/transforms.py
COPY ./models ./models
COPY ./app.py ./app.py
# Copy across pretrained classifier and backbone
COPY ./saved_models/model_20240328_074804_4 ./saved_models/model_20240328_074804_4
COPY ./pretrained_models/*.pth ${TORCH_HOME}/hub/
# Copy across requirements
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_pytorch.txt ./requirements_pytorch.txt

# -- Install Requirements --
# Two requirements files are ... required... 
# One for non-PyTorch modules and the other for PyTorch modules.
# This is because, by default, the GPU version of the PyTorch libraries are
# installed (along with the CUDA libraries). To create a container without
# the CUDA libraries, one must use the `--index-url` option. However, this
# applies globally to the `pip3 install` run so pip may be looking in the
# PyTorch repository for the other requirements.
RUN pip3 install -r ./requirements.txt 
RUN pip3 install -r ./requirements_pytorch.txt --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8501/tcp

ENTRYPOINT [ "streamlit", "run", "./app.py"]
