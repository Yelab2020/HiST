FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y openslide-tools

RUN conda install -y -c conda-forge python=3.8 python-spams=2.6.1 opencv rpy2

RUN pip install numpy==1.22 imgaug albumentations pandas matplotlib openslide-python scikit-learn staintools lifelines openpyxl palettable leidenalg ipykernel tqdm 'scanpy[leiden]'

COPY ./resource/timm-0.5.4.tar /app/resource/timm-0.5.4.tar

RUN pip install ./resource/timm-0.5.4.tar

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libblas3 liblapack3 libatlas-base-dev libgfortran5
