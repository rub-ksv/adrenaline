# Attention-based Deep Recurrent Network for Localizing Acoustic Events (ADRENALINE)

This repository contains the codebase accompanying the publication:

> Christopher Schymura, Tsubasa Ochiai, Marc Delcroix, Keisuke Kinoshita, Tomohiro Nakatani, Shoko Araki, Dorothea Kolossa, "Exploiting Attention-based Sequence-to-Sequence Architectures for Sound Event Localization", *European Signal Processing Conference (EUSIPCO 2020)*

\[ [IEEEXplore](https://ieeexplore.ieee.org/document/9287224) \]

## Summary

The proposed ADRENALINE architecture is shown below. The box on the left shows an exemplary encoding process for three discrete time-steps. A CNN-based feature extraction stage similar to the one used in the [SELDnet architecture](https://github.com/sharathadavanne/seld-net) is exploited to derive features from raw multi-channel audio signals. The attention weights are computed via scaled dot products between the encoder hidden states and the corresponding decoder hidden state from the previous decoding time-step. A context vector is derived as a weighted sum of the encoder hidden states, using the attention weights. The output of the decoder is composed of the source activity indicator and the corresponding source directions-of-arrival (DoAs), comprising azimuth and elevation. A concatenation of the decoder output from the previous time-step and the current context vector serves as input to the decoder, as shown in the box on the right.

<div align="center">
   <img src="./images/architecture.png" width="800" title="ADRENALINE architecture">
<p>Overview of the general ADRENALINE architecture.</p>
</div>

## Getting started

1.  Checkout the repository and download the required datasets via `$ ./download_data.sh`.

2.  It is recommended to use a dedicated virual environment for running the code:
    >  `$ sudo apt install virtualenv`\
    >  `$ virtualenv --python=python3.7 ./venv`\
    >  `$ source ./venv/bin/activate`\
    >  `$ pip install -r requirements.txt`

3.  Start an experiment via the main script `run.py` by selecting a specific configuration file (e.g. the CNN baseline model `cnn.yaml`):
    >  `$ python run.py --config ./configs/cnn.yaml --data_root /path/to/datasets`

    For further details on arguments accepted by the main script, type `python run.py --help`.

## Display training progress and test results
    
You can display the training progress and results on the validation and test sets using TensorBoard. In the default setting, evaluation log-files will be stored in a folder `./experiments`. Simply type
>  `$ tensorboard --logdir experiments`

to start a TensorBoard instance, which shows all tracked training, validation and test parameters.
