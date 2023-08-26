docker run \
    -it \
    --ipc=host \
    -p=8000:8000 \
    --name="monailabel_060" \
    -v $(pwd)/:/segmentation_middle_ear/ \
    projectmonai/monailabel:0.6.0 \
    /bin/bash

