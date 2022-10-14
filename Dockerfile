FROM jupyter/scipy-notebook

ENV HOME=/home/jovyan
WORKDIR $HOME

RUN pip install git+https://github.com/domna/SpectroscopicMapping