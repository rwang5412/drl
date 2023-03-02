echo enter name for new conda environment
read varname
#build conda env
conda env create --name $varname --file environment.yaml
eval "$(conda shell.bash hook)"
#set solver to use libmamba
conda config --set solver libmamba
#activate conda env
conda activate $varname
if ! test -d ~/.mujoco/mujoco210; then
    echo "Installing Mujoco210."
    pwd="$PWD"
    echo installing mujoco v2.10
    mkdir -p ~/.mujoco
    cd ~/.mujoco 
    rm -rf mujoco210-linux-x86_64.tar.gz
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    tar -xvzf mujoco210-linux-x86_64.tar.gz
    rm -rf mujoco210-linux-x86_64.tar.gz
    cd $pwd
fi 