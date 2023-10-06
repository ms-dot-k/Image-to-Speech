import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.IM_Speech import Im2Speech
from src.lr_scheduler import LinearWarmup, CosineAnnealingLRWarmup
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.dataset_IM import UnitDataset
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import torch.distributed as dist
import time
import glob
import wandb
import editdistance
from datetime import datetime
import sacrebleu, json
import soundfile as sf
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_co', default="data_to/COCO_2014")
    parser.add_argument('--speech_unit_path_co', default="data_to/SpokenCOCO/Hubert_units")
    parser.add_argument('--split_path_co', default="data_to/SpokenCOCO")
    parser.add_argument('--image_path_fl', default="data_to/Flickr8k/Images")
    parser.add_argument('--speech_unit_path_fl', default="data_to/flickr_audio/Hubert_units")
    parser.add_argument('--split_path_karpathy', default="data_to/Karpathy_split")

    parser.add_argument("--num_sp_unit", type=int, default=200)
    parser.add_argument("--max_sp_len", type=int, default=1024)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int, default=14)

    parser.add_argument("--temp_dir", type=str, default='./tmp_eval/IM_Speech')
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/IM_Speech')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--update_frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--tot_iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--warmup", default=False, action='store_true')
    parser.add_argument("--warmup_iteration", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=5000)

    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--mode", type=str, default='train', help='train, test, valid')

    parser.add_argument("--architecture", default='git-large')
    parser.add_argument("--vit_fix", default=False, action='store_true')

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

    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    train_data = UnitDataset(
        image_path=[args.image_path_co, args.image_path_fl],
        speech_unit_path=[args.speech_unit_path_co, args.speech_unit_path_fl],
        split_path=[args.split_path_co, args.split_path_karpathy],
        mode=args.mode,
        num_sp_unit=args.num_sp_unit,
        max_sp_len=args.max_sp_len
    )

    model = Im2Speech(args, train_data.num_sp_unit)
    num_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint

    model.cuda()

    if not args.vit_fix:
        params = [{'params': model.parameters()}]
    else:
        params = []
        for k, v in model.named_parameters():
            if not 'git.image_encoder' in k and not 'git.visual_projection' in k:
                params += [{'params': v}]
            else:
                v.requires_grad = False
    
    if not args.vit_fix:
        num_train = sum(p.numel() for p in params[0]['params'])
    else:
        num_train = sum(p['params'].numel() for p in params)
    if args.local_rank == 0:
        print(f'Train # of params: {num_train} / {num_model}')

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup:
        if args.tot_iters is not None:
            scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.tot_iters, T_warmup=args.warmup_iteration)
        else:
            scheduler = LinearWarmup(optimizer, T_warmup=args.warmup_iteration)
    else:
        scheduler = None

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif args.dataparallel:
        model = DP(model)

    # _ = validate(model, fast_validate=True)
    train(model, train_data, args.epochs, optimizer=optimizer, scheduler=scheduler, args=args)

def train(model, train_data, epochs, optimizer, scheduler, args):
    best_val_bleu = 0.0
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])
        if args.project is not None:
            wandbrun = wandb.init(project="Im2Speech", name=args.project + f'_{dt_string}')
            wandbrun.config.epochs = args.epochs
            wandbrun.config.batch_size = args.batch_size
            wandbrun.config.learning_rate = args.lr
            wandbrun.config.architecture = args.architecture
            wandbrun.config.vit_fix = args.vit_fix
            wandbrun.config.eval_step = args.eval_step
            wandbrun.config.update_frequency = args.update_frequency
            wandbrun.config.warmup = args.warmup
            wandbrun.config.warmup_iteration = args.warmup_iteration
        else:
            wandbrun = None
    else:
        writer = None
        wandbrun = None

    model.train()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda x: train_data.collate_fn(x),
    )

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        uer_list = []
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            prev_time = time.time()
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %f ********" % (
                    epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['lr']))
            im_unit, sp_unit, sp_unit_len, _ = batch

            output = model(im_unit.cuda(), sp_unit.cuda(), sp_unit_len.cuda(), inference=False)

            results = F.softmax(output.logits, dim=2).cpu()
            _, results = results.topk(1, dim=2)
            results = results.squeeze(dim=2).detach().numpy()

            loss = output.loss / args.update_frequency

            loss.backward()
            if ((i + 1) % args.update_frequency == 0) or (i + 1 == len(dataloader)):
                step += 1
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            else:
                continue

            sp_unit = sp_unit[:, 1:]
            pred_sp_unit = [" ".join([str(u) for u in r]) for r in results]
            gt_sp_unit = [" ".join([str(u) for u in r.numpy()]) for r in sp_unit]
            pred_sp_unit = [pred_sp[:pred_sp.find(f" {train_data.eos} ")] for pred_sp in pred_sp_unit]
            gt_sp_unit = [gt_sp[:gt_sp.find(f" {train_data.eos} ")] for gt_sp in gt_sp_unit]

            uer = uer_calc(pred_sp_unit, gt_sp_unit)

            loss_list.append(loss.cpu().item())
            uer_list.extend(uer)

            if args.local_rank == 0 and writer is not None:
                writer.add_scalar('train/loss', loss.cpu().item(), step)
                writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]['lr'], step)
                if i % 100 == 0:
                    print(f'######## Step(Epoch): {step}({epoch}), Loss: {loss.cpu().item()} #########')
                    for (predict, truth) in list(zip(pred_sp_unit, gt_sp_unit))[:3]:
                        print(f'PR: {predict}')
                        print(f'GT: {truth}\n')
                    writer.add_scalar('train/uer', np.array(uer_list).mean(), step)
                    if wandbrun is not None:
                        wandbrun.log({'train/loss': loss.cpu().item()}, step)
                        wandbrun.log({'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
                        wandbrun.log({'train/uer': np.array(uer_list).mean()}, step)

            if step % args.eval_step == 0:
                logs = validate(model, epoch=epoch, writer=writer, fast_validate=True, wandbrun=wandbrun, step=step)

                if args.local_rank == 0:
                    print('VAL_UER: ', logs[0])
                    print('VAL_BLEU: ', logs[1])
                    print('Saving checkpoint: %d' % epoch)
                    if args.dataparallel or args.distributed:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({'state_dict': state_dict},
                               os.path.join(args.checkpoint_dir, 'Epoch_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[1])))

                    if logs[1] > best_val_bleu:
                        best_val_bleu = logs[1]
                        bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                        for prev in bests:
                            os.remove(prev)
                        torch.save({'state_dict': state_dict},
                                   os.path.join(args.checkpoint_dir, 'Best_%04d_%05d_%.2f.ckpt' % (epoch, step, logs[1])))
            
            if args.tot_iters is not None and step == args.tot_iters:
                if args.local_rank == 0:
                    if args.dataparallel or args.distributed:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    torch.save({'state_dict': state_dict},
                                os.path.join(args.checkpoint_dir, 'Last_%05d.ckpt' % (step)))
                if args.distributed:
                    dist.barrier()
                assert 1 == 0, "Finishing Training, arrived total iterations"

    if args.local_rank == 0:
        print('Finishing training')


def validate(model, fast_validate=False, epoch=0, writer=None, wandbrun=None, step=0):
    with torch.no_grad():
        model.eval()

        val_data = UnitDataset(
            image_path=[args.image_path_co, args.image_path_fl],
            speech_unit_path=[args.speech_unit_path_co, args.speech_unit_path_fl],
            split_path=[args.split_path_co, args.split_path_karpathy],
            mode='val',
            num_sp_unit=args.num_sp_unit,
            max_sp_len=args.max_sp_len
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=int(args.batch_size * 1.5),
            num_workers=args.workers,
            drop_last=False,
            collate_fn=lambda x: val_data.collate_fn(x),
        )

        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(10 * batch_size, int(len(dataloader.dataset)))
            max_batches = 10
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        uer_list = []

        if args.local_rank == 0:
            if os.path.exists(os.path.join(args.temp_dir, 'unit')):
                shutil.rmtree(args.temp_dir)

        description = 'Validation on subset of the Val dataset' if fast_validate else 'Validation'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            im_unit, sp_unit, sp_unit_len, im_names = batch

            output = model(im_unit.cuda(), sp_unit.cuda(), sp_unit_len.cuda(), inference=True)

            results = output[:, 1:].cpu().detach().numpy()
            sp_unit = sp_unit[:, 1:]

            pred_sp_unit = [" ".join([str(u) for u in r]) for r in results]
            gt_sp_unit = [" ".join([str(u) for u in r.numpy()]) for r in sp_unit]
            pred_sp_unit = [pred_sp[:pred_sp.find(f" {val_data.eos} ")] for pred_sp in pred_sp_unit]
            gt_sp_unit = [gt_sp[:gt_sp.find(f" {val_data.eos} ")] for gt_sp in gt_sp_unit]

            uer_list.extend(uer_calc(pred_sp_unit, gt_sp_unit))

            if args.local_rank == 0:
                if (i % 10) == 0:
                    for j in range(batch_size):
                        print(f"Label: {gt_sp_unit[j]}\nPred : {pred_sp_unit[j]}\n")

            pred_sp_unit = [val_data.del_special_room(np.array([int(u) for u in pred_sp.split(" ") if u != ""], dtype=np.int64)) for pred_sp in pred_sp_unit]
            
            if args.distributed:
                dist.barrier()
            if args.local_rank == 0:
                for pred_sp, f_name in zip(pred_sp_unit, im_names):
                    save_name = os.path.join(args.temp_dir, 'unit', f_name + '.unit')
                    if not os.path.exists(os.path.dirname(save_name)):
                        os.makedirs(os.path.dirname(save_name))
                    torch.save(pred_sp, save_name)
            if args.distributed:
                dist.barrier()

            if i >= max_batches:
                break

        if args.local_rank == 0 and writer is not None:
            writer.add_scalar('val/uer', np.mean(uer_list), epoch)
            if wandbrun is not None:
                wandbrun.log({'val/uer': np.mean(uer_list)}, step)

        ##### Wav Gen #####
        print('Generating WAV from Unit')
        from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
        with open('./Vocoder/config.json') as f:
            vocoder_cfg = json.load(f)
        vocoder = CodeHiFiGANVocoder('./Vocoder/g_00950000', vocoder_cfg).cuda()
        
        def load_code(in_file):
            unit_paths = glob.glob(f"{in_file}/*.unit")
            for unit_path in unit_paths:
                unit = torch.load(unit_path)
                yield unit_path, unit

        data = load_code(os.path.join(args.temp_dir, 'unit'))
        for d_path, d in tqdm(data):
            f_name = os.path.splitext(os.path.basename(d_path))[0]
            x = {
                "code": torch.LongTensor(d).view(1, -1).cuda(),
                }
            with torch.no_grad():
                wav = vocoder(x, True)
            if args.local_rank == 0:
                save_name = os.path.join(args.temp_dir, 'wav', f_name + '.wav')
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                sf.write(save_name, wav.detach().cpu().numpy(), 16000)
        
        if args.distributed:
            dist.barrier()
        del vocoder, x, wav

        ##### Txt Gen #####
        print('Generating Transcription from WAV')
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        asrmodel = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        asrmodel = asrmodel.cuda()

        def load_wav(in_file):
            wav_paths = glob.glob(f"{in_file}/*.wav")
            for wav_path in wav_paths:
                wav, sample_rate = sf.read(wav_path)
                if len(wav) < 16000:
                    wav = np.concatenate([wav, np.zeros([16000 - len(wav)])], axis=0)
                assert sample_rate == 16_000
                yield wav_path, wav, sample_rate

        data = load_wav(os.path.join(args.temp_dir, 'wav'))
        for d_path, d, sr in tqdm(data):
            f_name = os.path.splitext(os.path.basename(d_path))[0]
            inputs = processor(d, sampling_rate=sr, return_tensors="pt", padding="longest")
            with torch.no_grad():
                logits = asrmodel(inputs.input_values.cuda(), attention_mask=inputs.attention_mask.cuda()).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            assert len(transcription) == 1
            transcription = transcription[0]
            if args.local_rank == 0:
                save_name = os.path.join(args.temp_dir, 'transcription', f_name + '.txt')
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                with open(save_name, 'w') as f:
                    f.write(transcription)
        if args.distributed:
            dist.barrier()
        del asrmodel, logits, inputs

        ##### Score Gen #####
        print('Generating BLEU score')
        gt_lists = {}
        data = json.load(open(os.path.join(args.split_path_karpathy, 'dataset_coco.json')))
        for d in data['images']:
            if d['split'] == 'val':
                im_name = d['filename'][:-4]
                captions = []
                for c in d['sentences']:
                    captions.append(c['raw'].lower())
                gt_lists[im_name] = captions

        refs = []
        preds = []
        pred_files = glob.glob(os.path.join(args.temp_dir, 'transcription', '*.txt'))
        for p in pred_files:
            refs.append(gt_lists[os.path.basename(p)[:-4]])
            with open(p, 'r') as txt:
                try:
                    preds.append(txt.readlines()[0].strip().lower())               
                except:
                    preds.append(' ')             
        
        refs = list(zip(*refs))
        BLEU_score = sacrebleu.corpus_bleu(preds, refs).format()
        if args.local_rank == 0:
            print(BLEU_score)
        BLEU_score = float(BLEU_score.split()[2])

        model.train()

        if args.local_rank == 0:
            print('val_uer:', np.mean(uer_list), 'BLEU:', BLEU_score)
            if writer is not None:
                writer.add_scalar('val/bleu', BLEU_score, epoch)
                if wandbrun is not None:
                    wandbrun.log({'val/bleu': BLEU_score}, step)
        return np.mean(uer_list), BLEU_score

def uer_calc(predict, truth):
    uer = []
    for pred, truth in zip(predict, truth):
        uer.append(1.0 * editdistance.eval(pred.split(' '), truth.split(' ')) / len(truth.split(' ')))
    return uer

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

