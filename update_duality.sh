
echo enter name of host conda environment for duality
read varname
# Set the name of the Conda environment
ENV_NAME="$varname"

# Activate the Conda environment
conda activate "$ENV_NAME"

# Check if the environment is activated
if [[ $CONDA_DEFAULT_ENV != "$ENV_NAME" ]]; then
  echo "Error: Conda environment $ENV_NAME not found."
  exit 1
fi

PACKAGE_NAME="duality"
# Check if the package is installed
if pip show $PACKAGE_NAME >/dev/null 2>&1; then
  # Uninstall the package
  echo "Uninstalling previously installed version of $PACKAGE_NAME..."
  pip uninstall -y $PACKAGE_NAME
fi

echo "Conda environment $ENV_NAME is activated, updating Duality..."
#clone into duality, create .whl file and pip install duality
git clone git@github.com:osudrl/duality.git
cd duality
python setup.py bdist_wheel
pip install ./dist/*.whl
cd ..
rm -rf duality
exit 0