FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install sudo wget ffmpeg git libsm6 libxext6 libxrender-dev -y

# Install python dependencies
RUN pip install numpy opencv-python matplotlib imageio sklearn imageio-ffmpeg

WORKDIR /
RUN git clone https://github.com/wolterlw/hand_tracking.git
WORKDIR /hand_tracking
RUN git fetch
RUN ./load_models.sh
RUN git checkout multihand

# Add API
COPY api/ /api


# Create output directory
RUN mkdir /out

CMD /bin/bash



# docker build -t "google-pose" ~/pose/Google/
# docker run -it --rm -v /home/nlp/amit/somevideo.mp4:/video.mp4 google-pose /bin/bash