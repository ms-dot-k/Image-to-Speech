# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
from pathlib import Path
import random
import soundfile as sf
import torch

from tqdm import tqdm

from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

# from examples.speech_to_speech.generate_waveform_from_code import cli_main
from examples.speech_to_speech.preprocessing.data_utils import process_units

import os, glob
import numpy as np

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dump_result(args, unit_path, pred_wav, suffix=""):
    assert args.in_code_file in unit_path
    out_path = os.path.splitext(unit_path.replace(args.in_code_file, args.results_path))[0]+'.wav'
    # if not os.path.exists(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(
        out_path,
        pred_wav.detach().cpu().numpy(),
        16000,
    )

def load_code(in_file):
    unit_paths = glob.glob(f"{in_file}/**/*.unit", recursive=True)
    for unit_path in unit_paths:
        unit = torch.load(unit_path)
        unit = process_units(unit, reduce=True)
        yield unit_path, unit

def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
    if use_cuda:
        vocoder = vocoder.cuda()

    multispkr = vocoder.model.multispkr
    if multispkr:
        logger.info("multi-speaker vocoder")
        num_speakers = vocoder_cfg.get(
            "num_speakers", 200
        )  # following the default in codehifigan to set to 200
        assert (
            args.speaker_id < num_speakers
        ), f"invalid --speaker-id ({args.speaker_id}) with total #speakers = {num_speakers}"

    data = load_code(args.in_code_file)
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    # for i, d in tqdm(enumerate(data), total=len(data)):
    for d_path, d in tqdm(data):
        x = {
            "code": torch.LongTensor(d).view(1, -1),
        }
        suffix = ""
        if multispkr:
            spk = (
                random.randint(0, num_speakers - 1)
                if args.speaker_id == -1
                else args.speaker_id
            )
            suffix = f"_spk{spk}"
            x["spkr"] = torch.LongTensor([spk]).view(1, 1)

        x = utils.move_to_cuda(x) if use_cuda else x
        wav = vocoder(x, args.dur_prediction)
        dump_result(args, d_path, wav, suffix=suffix)
        # assert 1==0

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-code-file", type=str, required=True, help="one unit sequence per line"
    )
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg",
        type=str,
        required=True,
        help="path to the CodeHiFiGAN vocoder config",
    )
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=-1,
        help="Speaker id (for vocoder that supports multispeaker). Set to -1 to randomly sample speakers.",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
