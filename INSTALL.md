# Installation

To run the Jupyter notebooks on your machine, you first need to install git.

Next, you need to close this repository. You can open a terminal and type the following commands: 

    $ cd $HOME # or any directory you prefer`
    $ git clone https://github.com/rachida-sgh/auto-mpg.git
    $ cd auto-mpg

You can also download master.zip, unzip it, rename the directory to auto-mpg.

# Requirements

Only Python 3.6 is supported.
You can check which version you have using the following commands:

    $ python3 --version

## Create an environment with Anaconda or Miniconda 

    $ conda -create auto-mpg python=3.6
    $ conda activate auto-mpg


This creates a new Python 3.6 environment called `auto-mpg`. You can change the name of the environment but it is recommended to have the same name for both the project and the virtual environment.

## Install Requirements with pip

With the virtual environment activated, type the following command:

    $ pip install -r requirements.txt

# Install the project

With the virtual environment activated, type the following command:

    $ pip install -e .

#  Starting Jupyter

You can now start Jupyter. Simply type the following command:

    $ jupyter notebook notebooks

You should see Jupyter's tree view of the notebooks in your browser. If your browser doesn't open automatically, visit localhost:8888 and click on `index.ipynb`.