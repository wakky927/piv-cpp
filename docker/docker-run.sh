docker run -d --cap-add sys_ptrace -p127.0.0.1:2224:22 --name opencv-dev lfc/opencv-devenv-clion
ssh-keygen -f "$HOME/.ssh/known_hosts" -R localhost:2224