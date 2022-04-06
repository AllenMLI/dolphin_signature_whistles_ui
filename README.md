# dolphin_signature_whistles_ui

This repository contains UI code and can be run together with dolphin_signature_whistles repository. 

Move the files from src to dolphin_whistles' src directory, move weights to dolphin_whistles' root directory, and the app folder under data to dolphin_whistles' data directory. Then follows User Interface direction as below.


## SETUP

### Get this Codebase onto Your Computer

Clone this repository
1. Log into your github account in the terminal.
2. `git clone https://github.com/AllenMLI/dolphin_whistles.git`

OR

Download the zip file of this repository
1. Click on the green Code button
2. Select Download Zip
3. Unzip the code directly under:
   * WINDOWS: `C:\Users\<YOUR-USERNAME>\dolphin_whistles`
   * MAC or LINUX: `/home/<YOUR-USERNAME>/dolphin_whistles`

### Install Anaconda (if not already installed)

Install Anaconda using the instructions for your operating system: https://docs.anaconda.com/anaconda/install
* NOTE: make sure to check the “add to PATH” box during installation
* If you get an error about “Failed to create Anaconda menus”:
    * If you have other versions of python installed, uninstall them
    * Turn off your antivirus software while installing
    * If you have Java Development Kit installed, uninstall that

Verify Anaconda install:
* Open Anaconda Powershell Prompt and run:
    * `conda list`
        * If conda installed properly, you’ll see a list of installed packages and their versions
    * `python`
        * A python shell should open up and tell you your version of python
    * Type in `quit()` to exit python


### Conda Environment:

Open Anaconda Powershell Prompt and run these commands (type in “y” and hit enter/return each time it asks if you want to proceed)
* NOTE: can’t use ^C/^V to copy/paste into Anaconda Prompt and right clicking also doesn’t seem to be an option, so need to type out each command

1) `conda create --name dolphin-env python=3.8`
2) `conda activate dolphin-env`
3) `conda install wandb`  ( If error occurs, try: `conda install -c conda-forge wandb`)
4) `conda install matplotlib`
5) `conda install -c conda-forge librosa`
6) `conda install git`
7) `pip install tensorflow`
8) `pip install opencv-python`
9) `pip install streamlit`
10) `pip install pip install awesome-streamlit`

Optionally, you may also install using the environment.yml file
`conda env create -f environment.yml`. This will install a virtual environment named "dolphin-whistles"

## Running Code

### User Interface

In `app/functions/app_raven_classify.py` and `app/functions/app_classify.py`, replace the line
`classes = np.sort(['INSERT_CLASS1', 'INSERT_CLASS2', 'INSERT_CLASS3'])`
with the actual class names.

To use the graphical user interface, please checkout AllenMLI's UI or AI2 Skiff's repository, install the additional packages as instructed, and place the app.py file and app folder in side the dolphin directory, run the below commands.
1) To run the user interface, make sure your conda environment is activated.
2) `cd` into dolphin_whistles
3) Run the following:
   * Windows: `streamlit run .\src\dolphin\app.py --server.maxUploadSize 1000`
   * Linux or Mac: `streamlit run src/dolphin/app.py --server.maxUploadSize 1000`
4) The interface should automatically open up in your browser.
5) IF an error pops up, complaining about a port already being used then run the following with a random number instead of 44:
   * Windows: `streamlit run .\src\dolphin\app.py --server.maxUploadSize 1000 --server.port=44`
   * Linux or Mac: `streamlit run src/dolphin/app.py --server.maxUploadSize 1000 --server.port=44`

To backup your environment,

`conda env export > environment.yml`
