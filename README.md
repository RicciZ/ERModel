## Environment
```bash
# python env
conda create -n ER
conda activate ER
# 3rd-party libs
conda install -c anaconda tqdm
# torch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
# use cuda11 in our server, so change this accordingly
# install if needed
conda install -c conda-forge h5py
conda install -c conda-forge plyfile
conda install -c anaconda scipy
conda install -c conda-forge pandas
conda install -c anaconda networkx
```

## Server
### open notebook on the server via browser
```bash
# 1. open notebook on the server
#    --ip=0.0.0.0 means visit with any ip
#    --no-browser means open without browser
#    --port=XXXX open with specific port you want
jupyter notebook --ip=0.0.0.0 --no-browser --port=XXXX

# 2. connect local port YYYY to the server port XXXX
ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost

# 3. visit localhost:YYYY via broser locally
```

## TMUX
|      shortcut key & command            |      function                                |
|:--------------------------------------:|:--------------------------------------------:|
| Ctrl + b + [                           | to see history (q to exit)                   |
| Ctrl + b + d                           | detach                                       |
| Ctrl + d                               | directly exit and kill current session       |
| Ctrl + b + s                           | list the sessions                            |
| Ctrl + b + $                           | rename current session                       |
| Ctrl + b + %                           | split window left and right                  |
| Ctrl + b + "                           | split window up and down                     |
| Ctrl + b + <arrow key>                 | switch cursor to other panes                 |
| Ctrl + b + x                           | close current pane                           |
| Ctrl + b + q                           | display pane number                          |
| Ctrl + b + $                           | rename current session                       |
| Ctrl + b + %                           | split window left and right                  |
| tmux ls                                | see the tmux info                            |
| tmux new -s <session-name>             | create session                               |
| tmux a -t <session-name>               | attach to a session                          |
| tmux kill-session -t <session-name>    | kill a session                               |




## Train
### GPU Use
```bash
# check GPU info
nvidia-smi
# Specify the GPU we want to use 
# e.g. to use GPU no. 2 to run train.py
CUDA_VISIBLE_DEVICES=2 python train.py
```


