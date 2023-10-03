import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import argparse
import json
import logging
from pathlib import Path
import random
import soundfile as sf
import torch

from tqdm import tqdm

from fairseq import utils

import os, glob

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, wav_path, transcription):
    assert args.in_wav_file in wav_path
    out_path = os.path.splitext(wav_path.replace(args.in_wav_file, args.results_path))[0] + '.txt'
    # if not os.path.exists(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(transcription)


import resampy


def load_wav(in_file):
    wav_paths = glob.glob(f"{in_file}/**/*.wav", recursive=True)
    for wav_path in wav_paths:
        wav, sample_rate = sf.read(wav_path)

        if sample_rate != 16_000:
            wav = resampy.resample(wav, sample_rate, 16_000)
            sample_rate = 16_000

        assert sample_rate == 16_000
        yield wav_path, wav, sample_rate


def main(args):
    logger.info(args)

    if args.language == "en":
        model_path = "facebook/wav2vec2-large-960h-lv60-self"
    elif args.language == "es":
        model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    elif args.language == "fr":
        model_path = "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french"
    elif args.language == "pt":
        model_path = "jonatasgrosman/wav2vec2-xls-r-1b-portuguese"
    elif args.language == "it":
        model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
    elif args.language == "nl":
        model_path = "jonatasgrosman/wav2vec2-xls-r-1b-dutch"
    elif args.language == "de":
        model_path = "jonatasgrosman/wav2vec2-xls-r-1b-german"
    elif args.language == "el":
        model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-greek"
    elif args.language == "ru":
        model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"

    processor = Wav2Vec2Processor.from_pretrained(model_path)
    # processor = utils.move_to_cuda(processor)

    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model = model.cuda()
    # model = utils.move_to_cuda(model)

    data = load_wav(args.in_wav_file)
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    # for i, d in tqdm(enumerate(data), total=len(data)):
    for d_path, d, sr in tqdm(data):
        # d = utils.move_to_cuda(d)
        inputs = processor(d, sampling_rate=sr, return_tensors="pt", padding="longest")

        with torch.no_grad():
            # logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
            logits = model(inputs.input_values.cuda(), attention_mask=inputs.attention_mask.cuda()).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        assert len(transcription) == 1
        transcription = transcription[0]

        dump_result(args, d_path, transcription)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-wav-file", type=str, required=True, help="one unit sequence per line"
    )
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()