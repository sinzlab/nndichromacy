FROM sinzlab/pytorch:latest

ADD . /src/nnfabrik
WORKDIR /src

RUN pip3 install -e nnfabrik
RUN pip3 install -e nnfabrik/ml-utils
RUN pip3 install -e nnfabrik/nndichromacy/nndichromacy
RUN pip3 install -e nnfabrik/mei/mei
RUN pip3 install -e nnfabrik/dataport/data_port

WORKDIR /notebooks