NAME=decoding-diffusion-$(date +%s)
echo New as : 
echo $NAME
echo Killed :
docker ps -aq --format '{{.Names}}' | grep '^decoding-diffusion-' | xargs -r docker stop
docker build -t $NAME-img . && \
docker run \
    $1 \
    -it --rm \
    -v $(pwd):/workspace \
    --gpus all \
    --name $NAME \
    --ipc=host \
    $NAME-img \
    /bin/bash
