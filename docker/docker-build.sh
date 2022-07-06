cd $(dirname $0)
docker build -t lfc/opencv-devenv -f ./Dockerfile.devenv .
docker build -t lfc/opencv-devenv-clion -f ./Dockerfile.devenv-clion .