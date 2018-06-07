# Face authorization

Authorization via face detection in python using some Machine Learning approaches and image processing

## dependencies

 - https://github.com/natanielruiz/deep-head-pose 
 - https://github.com/mpatacchiola/deepgaze (python 2.7 with opencv 2.x !!!)
 - 


### using conda

 - install anaconda https://conda.io/docs/user-guide/install/linux.html
     - download the Anaconda installer for Linux and follow the steps
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
    export PATH="$HOME/anaconda/bin:$PATH"
    conda install pytorch torchvision -c pytorch
    conda install -c menpo opencv3
    ```