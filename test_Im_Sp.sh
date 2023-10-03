#!/bin/sh
ROOT=./evaluation/Image_to_Speech
UNIT=${ROOT}/unit
WAV=${ROOT}/wav
TRANS=${ROOT}/transcription
DEVICE_ID=0

REF=data_to/COCO_2014
CKPT=./pretrained/IM_to_SP.ckpt
Fairseq_path=path_to/fairseq

python test_Im_Sp.py \
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