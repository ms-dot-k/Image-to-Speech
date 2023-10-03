# Towards Practical and Efficient Image-to-Speech Captioning with Vision-Language Pre-training and Multi-modal Tokens

This repository contains the PyTorch implementation of the following paper:
> **Towards Practical and Efficient Image-to-Speech Captioning with Vision-Language Pre-training and Multi-modal Tokens**<br>
> Minsu Kim, Jeongsoo Choi, Soumi Maiti, Jeong Hun Yeo, Shinji Watanabe, and Yong Man Ro<br>
> \[[Paper](https://arxiv.org/abs/2309.08531)\] \[[Project](https://ms-dot-k.github.io/Image-to-Speech-Captioning/)\]

<div align="center"><img width="30%" src="img/Img.PNG?raw=true" /></div>

## Requirements
- python 3.8
- pytorch 1.12~
- ffmpeg
- tensorboard
- opencv-python
- pillow
- librosa
- editdistance
- transformers
- timm
- einops
- fairseq
- https://github.com/thuanz123/enhancing-transformers

### Datasets
#### Download
SpokenCOCO dataset
- https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi

Flickr8k Audio dataset
- https://groups.csail.mit.edu/sls/downloads/flickraudio/

COCO2014 dataset
- https://cocodataset.org/

Flickr8k dataset
- https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names

Karpathy split
- https://cs.stanford.edu/people/karpathy/deepimagesent/

## Testing the Model
To test the model, run following command:
```shell
# test example on LRS2
python test.py \
--data 'data_directory_path' \
--data_name 'LRS2'
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 20 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- Refer to `test.py` for the other parameters

## Pre-trained model checkpoints
The pre-trained ASR models for output-level content supervision and lip-to-speech synthesis models on LRS2 and LRS3 are available. <br>

| Model |       Dataset       |   STOI   |
|:-------------------:|:-------------------:|:--------:|
|ASR|LRS2 |   [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EYjyTk0Bxy9CqLVmshqVXWEBlZc2Tq_4JnC4ox1tQ7jXOA?e=s8rZMW)  |
|ASR|LRS3 |   [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EcPkEXJ9UgNInxbJX_eh5aYBoZDLnxMY8AAEDNEiyBEJjw?e=uytxOK)  |
|Lip2Speech|LRS2 |   [0.526](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EWD7vxY4S7pPjNE8dUwSMJwBdgPFunw62HsDLIuUlWcKAQ?e=XYdHfn)  |
|Lip2Speech|LRS3 |   [0.497](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/Ea9mi0aKAa1Gu53jTKiQV0IB6x7s2rI1mG9hkgBdBCYWWg?e=SRcK6o)  |

## Training the Model
`data_name` argument is used to choose which dataset will be used. (LRS2 or LRS3) <br>
To train the model, run following command:

```shell
# Data Parallel training example using 2 GPUs on LRS2
python train.py \
--data '/data_dir_as_like/LRS2-BBC' \
--data_name 'LRS2'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 16 \
--epochs 200 \
--eval_step 3000 \
--dataparallel \
--gpu 0,1
```

```shell
# 1 GPU training example on LRS3
python train.py \
--data '/data_dir_as_like/LRS3-TED' \
--data_name 'LRS3'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 8 \
--epochs 200 \
--eval_step 3000 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--asr_checkpoint` : pretrained ASR checkpoint
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--output_content_on`: when the output content supervision is turned on (reconstruction loss)
- Refer to `train.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy time costs of waveform conversion (griffin-lim). <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation metrics, generated mel-spectrogram, and audio


## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2023towards,
  title={Towards Practical and Efficient Image-to-Speech Captioning with Vision-Language Pre-training and Multi-modal Tokens},
  author={Kim, Minsu and Choi, Jeongsoo and Maiti, Soumi and Yeo, Jeong Hun and Watanabe, Shinji and Ro, Yong Man},
  journal={arXiv preprint arXiv:2309.08531},
  year={2023}
}
```
