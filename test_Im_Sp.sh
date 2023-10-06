#!/bin/sh
ROOT=./evaluation/Image_to_Speech
UNIT=${ROOT}/unit
WAV=${ROOT}/wav
TRANS=${ROOT}/transcription
DEVICE_ID=0

REF=data_to/COCO_2014
CKPT=./pretrained/Im_to_Speech_Pretrained.ckpt
Fairseq_path=path_to/fairseq

Image_path_co=dir_to/COCO_2014
Speech_path_co=dir_to/SpokenCOCO/Hubert_units
Split_path_co=dir_to/SpokenCOCO
Image_path_fl=dir_to/Flickr8k/Images
Speech_path_fl=dir_to/Flickr8k_audio/Hubert_units
Split_path_kp=dir_to/Karpathy_split

python test_Im_Sp.py \
--image_path_co ${Image_path_co} \
--speech_unit_path_co ${Speech_path_co} \
--split_path_co ${Split_path_co} \
--image_path_fl ${Image_path_fl} \
--speech_unit_path_fl ${Speech_path_fl} \
--split_path_karpathy ${Split_path_kp} \
--architecture git-large \
--beam_size 1 \
--checkpoint ${CKPT} \
--save_dir ${UNIT} \
--gpu ${DEVICE_ID} \
--test_data coco

CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
PYTHONPATH=${Fairseq_path} \
python generate_waveform_from_code.py \
--in-code-file ${UNIT} \
--vocoder ./Vocoder/g_00950000 \
--vocoder-cfg ./Vocoder/config.json \
--results-path ${WAV} \
--dur-prediction

CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
PYTHONPATH=${Fairseq_path} \
python asr.py \
--in-wav-file ${WAV} \
--results-path ${TRANS} \
--language en

CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
python score.py \
--pred ${TRANS} \
--ref ${REF} 
