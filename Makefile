NAME=mppi_playground
VERSION=0.0.1
DOCKER_IMAGE_NAME=$(NAME):$(VERSION)
CONTAINER_NAME=$(NAME)
GPU_ID=all

build-gpu:
	docker build --platform=linux/arm64 -t $(DOCKER_IMAGE_NAME) -f docker/gpu/Dockerfile .

# build-cpu:
# 	docker build -t $(DOCKER_IMAGE_NAME) -f docker/cpu/Dockerfile .

bash-gpu:
	xhost +local:docker && \
	docker run -it \
		--gpus '"device=${GPU_ID}"' \
		-v ${PWD}/workspace \
		-v ${PWD}:/workspace/$(NAME) \
		-v ${HOME}:$(HOME) \
		--rm \
		--shm-size 10G \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY \
		-p 5900:5900 \
		--name $(CONTAINER_NAME)-bash \
		$(DOCKER_IMAGE_NAME) \
		bash

# bash-cpu:
# 	xhost +local:docker && \
# 	docker run -it \
# 		-v ${PWD}/workspace \
# 		-v ${PWD}:/workspace/$(NAME) \
# 		--rm \
# 		--shm-size 10G \
# 		-v /tmp/.X11-unix:/tmp/.X11-unix \
# 		-e DISPLAY \
# 		-e LIBGL_ALWAYS_SOFTWARE=1 \
# 		-p 5900:5900 \
# 		--name $(CONTAINER_NAME)-bash \
# 		$(DOCKER_IMAGE_NAME) \
# 		bash

clear:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)