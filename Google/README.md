# Google Mediapipe Hand Tracker
Hands tracking and pose estimation.

### Building from scratch:
```bash
docker build -t "google-pose" .
```
Currently, the default entrypoint is `/bin/bash`.

### Running a container:
For input, we need to mount a video (`/video.mp4`), and optionally an output directory (`/out`).

```bash
nvidia-docker run -it --rm
    -v /home/nlp/amit/some_video.mp4:/video.mp4 
    google-pose /bin/bash
```



```bash
nvidia-docker run -it --rm
    -v /home/nlp/amit/PhD/meta-scholar/utils/../datasets/SLCrawl/versions/SpreadTheSign/videos/20259_lt-lt_0.mp4:/video.mp4 
    -v /home/nlp/amit/pose/Google/api:/api
    google-pose /bin/bash
```



### Running Pose Estimation (for now, no real use)
```bash
pip install matplotlib
python /api/api.py
```