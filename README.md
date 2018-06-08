# Face authorization

Authorization via face detection in python using some Machine Learning approaches and image processing

## dependencies

 - https://github.com/natanielruiz/deep-head-pose 
 - https://github.com/mpatacchiola/deepgaze (python 2.7 with opencv 2.x !!!)
 - 


### using conda

 - install anaconda https://conda.io/docs/user-guide/install/linux.html
     - download the Anaconda installer for Linux and follow the steps
     - export PATH="$HOME/anaconda/bin:$PATH"
     -     
 - create a new conda environment `py35_pytorch`
    - ```conda create -n py35_pytorch python=3.5 anaconda -y```
    - activate/deactive the workspace:
``` 
# To activate this environment, use:
# > source activate py35_pytorch
#
# To deactivate an active environment, use:
# > source deactivate
$ source activate py35_pytorch
```
    - then install the following package:
    ```
    conda install pytorch torchvision -c pytorch
    conda install -c menpo opencv3
    ```
    
### using pip environment:

```
sudo apt-get install virtualenv
cd <into the repository>
virtualenv -p `which python2` py_env2
cd py_env2
source bin/activate
```


## dependencies of deep-head-pose:

download the models:
 - https://github.com/davisking/dlib-models
 - http://dlib.net/