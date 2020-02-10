Setup

OCI
Create at least VM.Standard1.8 sized instance
Oracle Linux Image

Docker installation

https://docs.docker.com/install/linux/docker-ce/centos/

Data
CIFAR dataset homepage: https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-10 download link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Docker commands

Build
docker build -t <tag> .

docker build -t cifar:1.0 .

Run 
docker run -i -v <HostSourcePath>:<TargetContainerMountPath> --shm-size <AllocatedSharedMemorySize> -t <ImageTag>

docker run -i -v /mnt/jblau-ai-datasets-filesystem/cifar-data/:/app/cifar-data --shm-size 10G -t cifar:1.0