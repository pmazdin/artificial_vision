#! /bin/bash
ENV_NAME=my_py_env
sudo apt-get install virtualenv

export PYTHONPATH=
#Then create and activate the virtual environment:
virtualenv ${ENV_NAME}
. ${ENV_NAME}/bin/activate
# install the package (-e will keep the installed package up to date)
pip install -r requirements.txt

cd etc
#wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ..
