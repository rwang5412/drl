read -p "choice [u]pgrade or [c]reate new conda env: " mode
echo $mode
if [ "$mode" == "u" ]; then
    conda config --set solver classic
    conda env update --file environment.yaml --prune
    exit
elif [ "$mode" == "c" ]; then
    echo enter name for new conda environment
    read varname
    #update conda
    conda update -n base conda
    #build conda env
    conda env create --name $varname --file environment.yaml
    eval "$(conda shell.bash hook)"
    #set solver to use libmamba
    #known bug with conda update https://github.com/ContinuumIO/anaconda-issues/issues/13123
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
else
    echo Choice not recognized.
fi
exit