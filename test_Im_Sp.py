import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.IM_Speech import Im2Speech
from src.lr_scheduler import LinearWarmup
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.dataset_IM import UnitDataset
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import time
import glob
from torch.autograd import grad
import wandb
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_co', default="data_to/COCO_2014")
    parser.add_argument('--speech_unit_path_co', default="data_to/SpokenCOCO/Hubert_units")
    parser.add_argument('--split_path_co', default="data_to/SpokenCOCO")
    parser.add_argument('--image_path_fl', default="data_to/Flickr8k/Images")
    parser.add_argument('--speech_unit_path_fl', default="data_to/flickr_audio/Hubert_units")
    parser.add_argument('--split_path_karpathy', default="data_to/Karpathy_split")
    parser.add_argument('--test_data', type=str, default='coco')

    parser.add_argument("--num_sp_unit", type=int, default=200)
    parser.add_argument("--max_sp_len", type=int, default=1024)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)

    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/IM_to_Speech_unit')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=1)

    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--warmup", default=False, action='store_true')
    parser.add_argument("--warmup_iteration", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=5000)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--augmentations", default=True)

    parser.add_argument("--mode", type=str, default='test', help='train, test, valid')

    parser.add_argument("--architecture", default='git-large')

    parser.add_argument("--distributed", default=False, action='store_true')
    parser.add_argument("--dataparallel", default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = '7488'

    test_data = UnitDataset(
        image_path=[args.image_path_co, args.image_path_fl],
        speech_unit_path=[args.speech_unit_path_co, args.speech_unit_path_fl],
        split_path=[args.split_path_co, args.split_path_karpathy],
        mode=args.mode,
        num_sp_unit=args.num_sp_unit,
        max_sp_len=args.max_sp_len,
        test_data=args.test_data
    )

    model = Im2Speech(args, test_data.num_sp_unit)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    model.cuda()

    test(model, test_data, fast_validate=False)

def test(model, test_data, fast_validate=False, epoch=0, writer=None, wandbrun=None):
    with torch.no_grad():
        model.eval()

        dataloader = DataLoader(
            test_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: test_data.collate_fn(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(10 * batch_size, int(len(dataloader.dataset)))
            max_batches = 10
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        description = 'Test on subset of the Test dataset' if fast_validate else 'Test'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Test : %d / %d ********" % ((i + 1) * batch_size, samples))
            im_unit, sp_unit, sp_unit_len, im_names = batch

            output = model(im_unit.cuda(), sp_unit.cuda(), sp_unit_len.cuda(), inference=True)

            results = output[:, 1:].cpu().detach().numpy()
            sp_unit = sp_unit[:, 1:]

            pred_sp_unit = [" ".join([str(u) for u in r]) for r in results]
            gt_sp_unit = [" ".join([str(u) for u in r.numpy()]) for r in sp_unit]
            pred_sp_unit = [pred_sp[:pred_sp.find(f" {test_data.eos} ")] for pred_sp in pred_sp_unit]
            gt_sp_unit = [gt_sp[:gt_sp.find(f" {test_data.eos} ")] for gt_sp in gt_sp_unit]

            pred_sp_unit = [test_data.del_special_room(np.array([int(u) for u in pred_sp.split(" ") if u != ""], dtype=np.int64)) for pred_sp in pred_sp_unit]

            if args.local_rank == 0:
                if (i % 10) == 0:
                    for j in range(batch_size):
                        print(f"Pred : {pred_sp_unit[j]}\n")

            for pred_sp, f_name in zip(pred_sp_unit, im_names):
                save_name = os.path.join(args.save_dir, f_name + '.unit')
                torch.save(pred_sp, save_name)

            if i >= max_batches:
                break

        if args.local_rank == 0:
            print('Finishing Testing')
        return

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

