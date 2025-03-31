#!/bin/bash
# reset_env.sh
# This script removes the existing "my_env" virtual environment if it exists,
# creates a new one, and installs the necessary packages for your project.

# Check if the "my_env" directory exists and remove it.
if [ -d "my_env" ]; then
    echo "Removing existing virtual environment 'my_env'..."
    rm -rf my_env
else
    echo "No existing virtual environment 'my_env' found."
fi

# Create a new virtual environment called my_env.
echo "Creating new virtual environment 'my_env'..."
python3 -m venv my_env

# Activate the new virtual environment.
echo "Activating virtual environment..."
source my_env/bin/activate

# Upgrade pip.
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages.
echo "Installing required packages..."
pip install torch transformers bitsandbytes datasets peft

echo "Environment reset complete. Your environment 'my_env' is ready."
echo "To activate it in the future, run: source my_env/bin/activate"
