FROM continuumio/miniconda:4.5.4

RUN pip install mlflow>=1.0 \
    && pip install numpy \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install keras