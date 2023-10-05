# Towards Practical and Efficient Image-to-Speech Captioning with Vision-Language Pre-training and Multi-modal Tokens

This repository contains the PyTorch implementation of the following paper:
> **Towards Practical and Efficient Image-to-Speech Captioning with Vision-Language Pre-training and Multi-modal Tokens**<br>
> Minsu Kim, Jeongsoo Choi, Soumi Maiti, Jeong Hun Yeo, Shinji Watanabe, and Yong Man Ro<br>
> \[[Paper](https://arxiv.org/abs/2309.08531)\] \[[Project](https://ms-dot-k.github.io/Image-to-Speech-Captioning/)\]

<div align="center"><img width="50%" src="img/Img.png?raw=true" /></div>

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

### Datasets Download
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

### Directory structure
```
COCO_2014
├── annotations
|   └── *.json
├── train2014
|   └── *.jpg
├── val2014
└── test2014

SpokenCOCO
├── *.json
└── Hubert_units
    ├── train
    |   └── *
    |       └── *.unit
    └── val

Flickr8k
├── Images
|   └── *.jpg
└── captions.txt

Flickr8k_audio
└── Hubert_units
    └── *.unit
```

## Extracting Speech Unit
We directly utilized the pre-trained K-means cluster model of [link](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit).
Please refer to the repository to extract speech unit (HuBERT Base + KM200).
The differences between the repository is the output format, we save each speech unit by using `torch.save(extracted_unit.cpu(), save_path + '.unit')`.
Please put the extracted units as the above directory structure.

## Image Unit Extractor
Please download `SeiT_weights.tar` from SeiT [github](https://github.com/naver-ai/seit/releases/tag/v0.0) and extract it.
Put `tokenizer.ckpt` and `codebook.ckpt` in `./pretrained/` directory.

## Testing the Model
To test the model, modify some argument in `test_Im_Sp.sh` and `test_Im_Sp_unit.sh`. <br>
Please refer below for the argument information.
After properly setting the argument, run following command:
```shell
# test example
sh test_Im_Sp.sh
```

Descriptions of important argument:
- `ROOT`: The output directory
- `DEVICE_ID`: GPU number
- `REF`: data dir path to COCO2014 (assuming the directory contains 'annotations/captions_val2014.json')
- `CKPT`: Model checkpoint
- `Fairseq_path`: The path for fairseq installed
- `Image_path_co`: data dir path to COCO2014 ('dir_to/COCO_2014')
- `Speech_path_co`: data dir path to Speech unit of SpokenCOCO ('dir_to/SpokenCOCO/Hubert_units')
- `Split_path_co`: data dir path to SpokenCOCO ('dir_to/SpokenCOCO', assuming the directory contains json files)
- `Image_path_fl`: data dir path to Flickr8k ('dir_to/Flickr8k/Images')
- `Speech_path_fl`: data dir path to Speech unit of Flickr8k ('dir_to/Flickr8k_audio/Hubert_units')
- `Split_path_kp`: data dir path to Karpathy split ('dir_to/Karpathy_split')


## Pre-trained model checkpoints
The pre-trained models are available. <br>

| Model |       Dataset       |   BLEU-4 (COCO)  |   BLEU-4 (Flickr 8k)  | Link |
|:-------------------:|:-------------------:|:--------:|:--------:|:--------:|
|Image to Speech| COCO & Flickr8k | 25.9 | 20.6 | [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EQHitBzytIxFocEk6acURjMB-QErSe29Z9gyhh-vl-srlQ?e=RZOw2p)  |
|Image unit to Speech| COCO & Flickr8k | 20.1 | 16.7 |  [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/Ec6Hq6VCk5dDjA5RUw7WYvcBbICcRMNC-pej-JeUgNKROg?e=DxnYpY)  |


## Training the Model
If you set `project` which will be the project name of wandb, you should have the wandb account and login.
To train the model, run following command:

```shell
# Distributed training example using 4 GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
train_Im_Sp.py \
--image_path_co dir_to/COCO_2014 \
--speech_unit_path_co dir_to/SpokenCOCO/Hubert_units \
--split_path_co dir_to/SpokenCOCO \
--image_path_fl dir_to/Flickr8k/Images \
--speech_unit_path_fl dir_to/Flickr8k_audio/Hubert_units \
--split_path_karpathy dir_to/Karpathy_split \
--checkpoint_dir ./data/checkpoints/IM_Speech \
--temp_dir ./tmp_eval/IM_Speech \
--project IM_Speech \
--architecture git-large-coco \
--batch_size 16 \
--eval_step 5000 \
--lr 5e-5 \
--gpu 0,1,2,3 \
--update_frequency 1 \
--start_epoch 0 \
--vit_fix \
--warmup \
--warmup_iteration 10000 \
--tot_iters 100000 \
--distributed \
```

Descriptions of training parameters are as follows:
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--eval_step`: steps to perform evaluation
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--update_frequency`: gradient accumulation steps
- `--vit_fix`: image encoder freeze
- Refer to `train_Im_Sp.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy inference time. <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).


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
