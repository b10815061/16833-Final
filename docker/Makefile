NR_JOBS ?= 8

build:
	docker compose build

run:
	xhost +local:docker
	docker compose run --rm dev

clean:
	docker compose down -v

# ── Mac targets (no GPU, non-root user 'dev') ─────────────────────────────────
mac-build:
	docker build -f Dockerfile.mac -t swarm-slam-mac .

mac-run:
	docker run --rm -it \
	    -v "$(shell pwd)/../src:/home/dev/ws/src/swarm_slam" \
	    -v "$(shell pwd)/../data:/home/dev/datasets:ro" \
	    swarm-slam-mac bash

.PHONY: build run clean mac-build mac-run
