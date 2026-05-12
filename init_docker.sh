set -ex

container_name='cybertron_model_align_nvidia'
image_name='modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/ngc-train:nvidia.a1e5b2ac'
docker run -itd --privileged \
    -v /home/test:/home/test \
    --cpus 80 \
    --memory 800g \
    --shm-size 800g \
    --net=host \
    --name ${container_name} \
    $image_name \
    /bin/bash
# docker exec -it -u root $container_name /bin/bash
docker exec -it $container_name /bin/bash
