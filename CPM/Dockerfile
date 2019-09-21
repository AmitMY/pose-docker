FROM nvcr.io/nvidia/tensorflow:19.08-py3

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

RUN git clone https://github.com/circulosmeos/gdown.pl.git gdown

RUN mkdir pose
WORKDIR /workspace/pose
RUN git clone https://github.com/timctho/convolutional-pose-machines-tensorflow.git cpm
RUN /workspace/gdown/gdown.pl 'https://drive.google.com/file/d/0Bx1hAYkcBwqnSU9lSm5Ya3B1VTg/view' 'cpm/models/model.pkl'
COPY api/ /workspace/pose/api

# docker build -t "cpm" .

# nvidia-docker run -it --rm -v /home/nlp/amit/pose/CPM/api:/workspace/pose/api -v /home/nlp/amit/pose/assets:/workspace/pose/assets -v /home/nlp/amit/pose/output:/workspace/pose/out cpm

# python api/hand.py --input assets/hands --output out --device cpu -f image