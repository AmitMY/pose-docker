# Google Mediapipe Hand Tracker
Hands tracking and pose estimation.

### Building from scratch:
```bash
docker build -t "google-pose" .
```
Currently, the default entrypoint is running pose estimation on a video, but you can use `/bin/bash`.

### Running a container:
For input, we need to mount a video (`/video.mp4`), and optionally an output directory (`/out`).

```bash
nvidia-docker run -it --rm
    -v /home/nlp/amit/some_video.mp4:/video.mp4 
    google-pose
```


### Running Pose Estimation
In case you want to manually execute the pose estimation command, in the docker run:
```bash
python /api/pose_video.py
```