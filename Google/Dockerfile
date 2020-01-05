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

CMD python /api/pose_video.py



# docker build -t "google-pose" ~/pose/Google/
# docker run -it --rm -v /home/nlp/amit/PhD/meta-scholar/utils/../datasets/SLCrawl/versions/SpreadTheSign/videos/20259_lt-lt_0.mp4:/video.mp4 google-pose /bin/bash