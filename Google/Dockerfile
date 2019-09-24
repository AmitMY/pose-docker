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

# Configure and build
RUN yes '' | ./configure
RUN bazel build --action_env PATH="$PATH" --noincompatible_strict_action_env --config=opt --incompatible_disable_deprecated_attr_params=false //tensorflow/tools/pip_package:build_pip_package
# Buid pip package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg

WORKDIR hand_tracking

# docker build -t "google-pose" .