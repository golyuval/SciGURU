FROM continuumio/anaconda3:4.10.3

WORKDIR /app

COPY ../../.. /app

RUN conda install -c pytorch pytorch
RUN pip install transformers

CMD ["python", "infer.py"]