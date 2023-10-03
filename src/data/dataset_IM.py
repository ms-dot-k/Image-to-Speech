import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
import json

class UnitDataset(Dataset):
    def __init__(self, image_path, speech_unit_path, split_path, mode, num_sp_unit=200, max_sp_len=1024, archictecture="git-large", test_data='coco'):
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.test_data = test_data
        self.im_paths, self.sp_paths = self.build_file_list(image_path, speech_unit_path, split_path, mode)

        self.bos = 101
        self.eos = 102
        self.pad = 0

        self.num_sp_unit = num_sp_unit + 3

        self.max_sp_len = max_sp_len
        self.processor = AutoProcessor.from_pretrained(f"microsoft/{archictecture}")

    def build_file_list(self, image_path, speech_unit_path, split_path, mode):
        im_files = []
        sp_files = []
        co_im, fl_im = image_path
        co_sp, fl_sp = speech_unit_path
        co_file, kp_file = split_path

        im2wav_mapping = {}
        data = json.load(open(os.path.join(co_file, f'SpokenCOCO_train.json'), 'r'))
        for d in data['data']:
            im_path = d['image'][:-4]    #train2014/~~.jpg or val2014/~~.jpg
            captions = d['captions']
            caps = []
            for cap in captions:
                cap = cap['wav'].replace('wavs/', '')[:-4]
                caps.append(cap)
            im2wav_mapping[im_path] = caps
        data = json.load(open(os.path.join(co_file, f'SpokenCOCO_val.json'), 'r'))
        for d in data['data']:
            im_path = d['image'][:-4]    #train2014/~~.jpg or val2014/~~.jpg
            captions = d['captions']
            caps = []
            for cap in captions:
                cap = cap['wav'].replace('wavs/', '')[:-4]
                caps.append(cap)
            im2wav_mapping[im_path] = caps

        if mode != 'test':
            data = json.load(open(os.path.join(kp_file, 'dataset_coco.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(co_im, d['filepath'], d['filename'])
                    captions = im2wav_mapping[f"{d['filepath']}/{d['filename'][:-4]}"]
                    if mode == 'train':
                        for cap in captions:
                            sp_path = os.path.join(co_sp, cap + '.unit')
                            im_files.append(im_path)
                            sp_files.append(sp_path)
                    else:
                        cap = captions[0]
                        sp_path = os.path.join(co_sp, cap + '.unit')
                        im_files.append(im_path)
                        sp_files.append(sp_path)

        elif self.test_data == 'coco':
            data = json.load(open(os.path.join(kp_file, 'dataset_coco.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(co_im, d['filepath'], d['filename'])
                    im_files.append(im_path)

        num_coco = len(im_files)
        print(f"num COCO {num_coco}")

        if mode == 'train':
            data = json.load(open(os.path.join(kp_file, 'dataset_flickr8k.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(fl_im, d['filename'])
                    for caption in range(5):
                        sp_path = os.path.join(fl_sp, 'flickr_audio/wavs', d['filename'][:-4] + f'_{caption}.unit')
                        if os.path.exists(sp_path):
                            im_files.append(im_path)
                            sp_files.append(sp_path)
        
        elif mode == 'test' and self.test_data == 'flickr':
            data = json.load(open(os.path.join(kp_file, 'dataset_flickr8k.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(fl_im, d['filename'])
                    im_files.append(im_path)

        print(f"num Flickr {len(im_files) - num_coco}")

        return im_files, sp_files

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        sp_unit = None
        if self.mode != 'test':
            sp_path = self.sp_paths[idx]
        skip = False

        image = Image.open(im_path).convert('RGB')

        im_input = self.processor(images=image, return_tensors="pt").pixel_values   #1, 3, 224, 224

        if self.mode != 'test':
            sp_unit = torch.load(sp_path)
            sp_unit = self.add_special_room(sp_unit)
            sp_unit = self.process_units(sp_unit, reduce=True)
            assert (sp_unit > self.num_sp_unit).sum() == 0

            sp_unit = self.append_bos(sp_unit)
            sp_unit = self.append_eos(sp_unit)

            if len(sp_unit) > self.max_sp_len:
                print(f'Skipping this sample due to long input length, sp:{len(sp_unit)}')
                skip = True

        return im_input, sp_unit, os.path.splitext(os.path.basename(im_path))[0], skip

    def process_units(self, units, reduce=False):
        if not reduce:
            return units
        out = [int(u) for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
        return torch.tensor(out)

    def add_special_room(self, units):
        # "microsoft/git-base" [PAD]:0 , [UNK]:100, [CLS]:101, [SEP]:102 [MASK]:103
        # CLS: BOS, SEP: EOS
        if self.bos == 101 and self.eos == 102:
            units[np.where(units>100)] += 2 #BOS, EOS room
            units += 1  #PAD room
            return units
        else:
            raise NotImplementedError

    def del_special_room(self, units):
        if self.bos == 101 and self.eos == 102:
            units -= 1 #Del PAD room
            units[np.where(units>100)] -= 2 #Del BOS, EOS room
            return units
        else:
            raise NotImplementedError

    def append_bos(self, units):
        return torch.cat([torch.tensor([self.bos]), units], 0)
    
    def append_eos(self, units):
        return torch.cat([units, torch.tensor([self.eos])], 0)

    def collate_fn(self, batch):
        sp_len, im_names = [], []
        for data in batch:
            if not data[3]:
                if self.mode != 'test':
                    sp_len.append(len(data[1]))
                im_names.append(data[2])

        if self.mode != 'test':
            max_sp_len = max(sp_len)

        im_inputs = []
        padded_sp_unit = []

        for i, (im_input, sp_unit, _, skip) in enumerate(batch):
            if not skip:
                im_inputs.append(im_input)
                if self.mode != 'test':
                    padded_sp_unit.append(torch.cat([sp_unit, torch.ones([max_sp_len - len(sp_unit)]) * self.pad], 0))

        im_inputs = torch.cat(im_inputs, 0)
        if self.mode != 'test':
            sp_len = torch.IntTensor(sp_len)
            sp_unit = torch.stack(padded_sp_unit, 0).long()
        else:
            sp_len = torch.ones(im_inputs.size(0)).long() * 512
            sp_unit = torch.ones([im_inputs.size(0), 512]).long()
        return im_inputs, sp_unit, sp_len, im_names
