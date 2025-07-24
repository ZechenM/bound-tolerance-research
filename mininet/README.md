# Mininet Setup Instructions

## Enter Mininet Docker Image

Make sure you have docker or docker desktop app installed. If you are using docker desktop app, make sure the you manually set the memory to a resonably large size under preferences/resources.

```bash
# Pull the image
docker pull davidlin123/mininet:latest

# Run with proper configuration
docker run -it --rm \
  --name mininet \
  --privileged \
  --network host \
  -v /lib/modules:/lib/modules \
  davidlin123/mininet:latest
```

This image is Ubuntu. So the packet manager is apt.

## Setup SSH

```bash
ssh-keygen
```

copy the public key to your github account and clone the this repo.

## Install Python 3.12

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

Index into project directory and create a virtual environment

```bash
cd bound-tolerance-research
which python3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

## Steps to run our application

1. in one bash session in the home directory:

```bash
./pox/pox.py forwarding.l2_learning
```

2. in another shell in the project home directory, run:

```bash
sudo python3 mininet/distributed-ml-topo.py
```

3. in four other shells in the project directory, run:

```bash
tail -F logs/server.log
tail -F logs/worker{i}.log
```
