# Emotion Recognition Model
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


## Train
To train the Emotion Recognition Model, run the following codes. 

    $ python main.py [-h] [--exp_name EXP_NAME] [--input_size INPUT_SIZE] 
                    [--num_layers NUM_LAYERS] [--hidden_size HIDDEN_SIZE] 
                    [--num_epochs NUM_EPOCHS] [--load_model LOAD_MODEL] 
                    [--lr LR] [--use_hrv USE_HRV]

        optional arguments:
            -h, --help                  show this help message and exit
            --exp_name EXP_NAME         Name of the experiment
            --input_size INPUT_SIZE     input size
            --num_layers NUM_LAYERS     num of layers for blstm
            --hidden_size HIDDEN_SIZE   hidden size for blstm
            --num_epochs NUM_EPOCHS     number of epochs
            --load_model LOAD_MODEL     load model or not
            --lr LR                     learning rate
            --use_hrv USE_HRV           use hrv info or not


## Server
To run the well trained Emotion Recognition Model on server, run the following codes. It will take input ECG file and output the corresponding emotion and recommended music types in a txt file. 

    $ python ERonServer.py [-h] [--ecg_file ECG_FILE]

        Emotion Recognition on Server

        optional arguments:
            -h, --help              show this help message and exit
            --ecg_file ECG_FILE     the path of the input ecg file




