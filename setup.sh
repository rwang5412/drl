echo enter name for new conda environment
read varname
#build conda env
conda create --name $varname --file spec-file.txt
eval "$(conda shell.bash hook)"
#set solver to use libmamba
conda config --set solver libmamba
#activate conda env
conda activate $varname
echo getting pip packages
#get pip requirements
pip install mujoco
pip install wandb
pip install ray==2.2.0
pip install setuptools==59.5.0
pip install ./sim/digit_sim/digit_ar_sim/agility-1.1.1-py3-none-any.whl

pwd="$PWD"
echo installing mujoco v2.10
mkdir -p ~/.mujoco
cd ~/.mujoco 
rm -rf mujoco210-linux-x86_64.tar.gz
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
rm -rf mujoco210-linux-x86_64.tar.gz
cd $pwd

