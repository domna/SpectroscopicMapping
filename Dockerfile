FROM jupyter/scipy-notebook

ENV JUPYTER_ENABLE_LAB yes
ENV HOME=/home/jovyan
ENV APP_HOME=/home/jovyan/SpectroscopicMapping
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

COPY --chown=${NB_UID}:${NB_GID} . $APP_HOME
RUN pip install --quiet --no-cache-dir 'flake8==3.9.2' -r requirements.txt
RUN python setup.py develop
