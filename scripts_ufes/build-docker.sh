#!/usr/bin/env bash
gpu=false
runfile=$2  #dqn_cartpole_batch.sh
project_path=$1

#TODO: Select cpu or cuda

if [ $gpu = true ];
then
  # Prepare docker with cuda: docker build -f docker/cuda.Dockerfile -t horizon:dev --memory=8g --memory-swap=8g .
  echo "Building image..."
  docker build -f "${project_path}"/ReAgent/docker/cuda.Dockerfile -t horizon:dev --memory=8g --memory-swap=8g .

  echo "Creating container..."
  docker run -d --gpus all --runtime=nvidia -v "${project_path}":/home/BatchRL -p 0.0.0.0:6006:6006 -t horizon:dev
else
    # Prepare docker with cuda: docker build -f docker/cuda.Dockerfile -t horizon:dev --memory=8g --memory-swap=8g .
  echo "Building image..."
  docker build -f "${project_path}"/ReAgent/docker/cpu.Dockerfile -t horizon:dev --memory=2g --memory-swap=1g .

  echo "Creating container..."
  docker run -d -v "${project_path}":/home/BatchRL -p 0.0.0.0:6006:6006 -t horizon:dev
fi

echo "Preparing container..."
conteiner_id=$(docker ps --latest --format "{{.ID}}")

# Only if will use gpu
#docker exec -t ${conteiner_id} bash -c "conda install -y pytorch cudatoolkit=10.1 -c pytorch-nightly"

docker exec -t "${conteiner_id}" sudo apt -y install jq moreutils
docker exec -t "${conteiner_id}" pip install matplotlib
docker exec -w /home/BatchRL -t "${conteiner_id}" bash -c "git submodule update --init --recursive && git submodule update --recursive"
docker exec -w /home/BatchRL/ReAgent -t "${conteiner_id}" bash -c "mvn -f preprocessing/pom.xml clean package"
docker exec -w /home/BatchRL/ReAgent -t "${conteiner_id}" bash -c "./scripts/setup.sh"

###
docker exec -w /home/BatchRL/scripts_ufes -t "${conteiner_id}" bash -c "bash ${runfile}"
