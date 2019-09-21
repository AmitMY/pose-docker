# CPM
This image currently only support hand pose estimation.

### Building from scratch:
```bash
docker build -t "cpm" .
```
Currently, the default entrypoint is `/bin/bash`.

### Running a container:
For input-output we need to mount at least two directories (one for input and one for output) and specify them in the executable.

```bash
nvidia-docker run -it --rm 
    -v /home/nlp/amit/pose/assets:/workspace/pose/assets
    -v /home/nlp/amit/pose/output:/workspace/pose/out
    cpm
```

### Running Pose Estimation
```bash
python api/hand.py --input assets/hands --output out
```
You can also specify the device to run on (gpu/cpu) and the output format (image/json).
