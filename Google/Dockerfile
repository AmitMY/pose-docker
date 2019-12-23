FROM python:3.7.4-stretch

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install sudo git wget build-essential unzip -y

RUN pip3 install numpy opencv-python

RUN git clone https://github.com/wolterlw/hand_tracking.git
RUN git clone https://github.com/google/mediapipe.git

# Clone Tensorflow correct version
RUN git clone https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
RUN git fetch origin tag v1.13.2
RUN git checkout v1.13.2
ENV TF_NEED_CUDA=0

# Copy the custom module from mediapipe to tensorflow
RUN mkdir /tensorflow/tensorflow/lite/python/custom_ops/
RUN cp -R /mediapipe/mediapipe/util/tflite/operations/. /tensorflow/tensorflow/lite/python/custom_ops/
RUN sed -i 's/mediapipe:__subpackages__/visibility:public/g' /tensorflow/tensorflow/lite/python/custom_ops/BUILD
RUN sed -i 's/@org_tensorflow//g' /tensorflow/tensorflow/lite/python/custom_ops/BUILD

# Install Bazel
ARG BAZEL_VERSION=0.21.0
RUN mkdir /bazel
RUN wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
RUN chmod +x /bazel/installer.sh
RUN /bazel/installer.sh
RUN rm -f /bazel/installer.sh

# Intall pip dependencies
RUN pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
RUN pip install -U --user keras_applications --no-deps
RUN pip install -U --user keras_preprocessing --no-deps
RUN pip install matplotlib

# Configure and build
RUN yes '' | ./configure
RUN bazel build --action_env PATH="$PATH" --noincompatible_strict_action_env --config=opt --incompatible_disable_deprecated_attr_params=false //tensorflow/tools/pip_package:build_pip_package
# Buid pip package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg
# Install pip package
RUN pip install /tensorflow/tensorflow_pkg/tensorflow-1.13.2-cp37-cp37m-linux_x86_64.whl

# Download the models
WORKDIR /hand_tracking
RUN mkdir models
RUN curl --header 'Host: raw.githubusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://github.com/google/mediapipe/blob/master/mediapipe/models/palm_detection.tflite' --header 'Upgrade-Insecure-Requests: 1' 'https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/models/palm_detection.tflite' --output './models/palm_detection.tflite'
RUN curl --header 'Host: raw.githubusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://github.com/google/mediapipe/blob/master/mediapipe/models/hand_landmark.tflite' --header 'Upgrade-Insecure-Requests: 1' 'https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/models/hand_landmark.tflite' --output './models/hand_landmark.tflite'

# docker build -t "google-pose" .