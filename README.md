# A Deep Learning Project

## Setup
build the docker container (with NVIDIA GPU)
`docker-compose build`

or
build the docker container (without NVIDIA GPU)
`docker-compose -f docker-compose-mac.yaml build`

## Development
run docker container (with NVIDIA GPU)
`docker-compose up`

or
build the docker container (without NVIDIA GPU)
`docker-compose -f docker-compose-mac.yaml up`

open the created jupyter link from console output
`http://127.0.0.1:8888/lab?token=...`


docker down for custom named docker-compose yml
`docker-compose -f docker-compose-mac.yaml down`