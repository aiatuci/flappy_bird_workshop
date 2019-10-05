# Flappy Bird Workshop
UCI AI Club Flappy Bird  

## Download

Use the "Clone or download" and click "Download ZIP".

Save the folder to your computer.

Open a Terminal (OSX/LINUX) or CMD (WINDOWS) and follow the below instructions.

## Prerequisites

We have provided a requirements.txt for you to setup your **python3.6.8 environment.**

Python 3.6.8 Download.
https://www.python.org/downloads/release/python-368/?fbclid=IwAR2tjkyGDbWLivj2TVRRBYpB5hSFk7LmKcCpCKJU9ASrmq2CNcdxwsSZKxg

**Has to be python3.6.8, other tensorflow will not install**

Change directories to flappy bird folder
```
cd /path/to/flappy/folder/
```

First off, create your virtual environment by entering the below command

### Mac OSX / Linux

If on OSX or Linux, enter the following into terminal
```
python3.6 -m venv flappyenv

source flappyenv/bin/activate

pip install -r requirements.txt
```

### Windows

If on windows, you may have to run the following in order to install the virtual environment tool

```
pip3.6 install virtualenv
```

Then, you have to run the following to make and activate the venv
```
virtualenv flappyenv

flappyenv\Scripts\activate.bat

pip install -r requirements.txt

```
### RUN

Once you have got your venv activated, run the following command

```
python flappy.py
```


