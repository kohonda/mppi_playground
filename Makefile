NAME=mppi_playground
VERSION=0.0.1
DOCKER_IMAGE_NAME=$(NAME):$(VERSION)
CONTAINER_NAME=$(NAME)
GPU_ID=all

build:
	docker build -t $(DOCKER_IMAGE_NAME) .

bash:
	xhost +local:docker && \
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/workspace \
		-v ${PWD}:/workspace/$(NAME) \
		--rm \
		--shm-size 10G \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY \
		-p 5900:5900 \
		--name $(CONTAINER_NAME)-bash \
		$(DOCKER_IMAGE_NAME) \
		bash

bash-wo-gpu:
	docker run -it \
		-v ${PWD}/workspace \
		-v ${PWD}:/workspace/$(NAME) \
		--rm \
		--shm-size 10G \
		--name $(CONTAINER_NAME)-bash \
		$(DOCKER_IMAGE_NAME) \
		bash
