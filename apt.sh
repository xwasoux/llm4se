sudo chmod -R 777 .
sudo apt update -y
sudo apt -y install \
    tig \
    htop \
    nvtop \
    tmux \
    vim \
    tree

echo "Enter your git account name: "
read user
echo "Enter your git account email: "
read email
git config --global user.name $user
git config --global user.email $email