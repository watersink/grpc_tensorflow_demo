#docker pull nvcr.io/nvidia/tritonserver:20.03-py3
docker run --gpus '"device=7"' --name="jxl-tritonserver" -d --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8020:8000 -p8021:8001 -p8022:8002 -v /home/jiangxiaolong/triton-inference-server:/models  nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-repository=/models
#curl localhost:8000/api/status
