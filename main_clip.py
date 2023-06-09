from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, Flowers102, CIFAR10, Food101, EuroSAT

import clip
from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname

import pandas as pd
import gc

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    return args


best_acc1 = 0
device = "cuda:{}" if torch.cuda.is_available() else "cpu"


def train_val_test_dataset(dataset, test_split, val_split):
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    train_idx, test_idx = train_test_split(all_indices, test_size=test_split)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_split)
    
    datasets = {}
    
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    datasets["test"] = Subset(dataset, test_idx)

    return datasets


def main():
    global best_acc1, device

    args = parse_option()
    args_pretty_print = []
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        args_pretty_print.append(f"{arg}: {getattr(args, arg)}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    device = device.format(args.gpu) if args.gpu != -1 else "cpu"

    # create model
    if args.arch == "vit_b32":
        model, preprocess = clip.load('ViT-B/32', device, jit=False)
    elif args.arch == "vit_b16":
        model, preprocess = clip.load('ViT-B/16', device, jit=False)
    convert_models_to_fp32(model)

    model.eval()
    prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                if isinstance(best_acc1, float):
                    best_acc1 = torch.tensor(best_acc1).float()
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')

    if args.dataset == "cifar100":
      train_dataset = CIFAR100(args.root, transform=preprocess,
                              download=True, train=True)

      val_dataset = CIFAR100(args.root, transform=preprocess,
                            download=True, train=False)
      
      test_dataset = val_dataset
    
    elif args.dataset == "cifar10":
        train_dataset = CIFAR10(args.root, transform=preprocess,
                              download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                            download=True, train=False)
        
        test_dataset = val_dataset

    elif args.dataset == "flowers102":
      train_dataset = Flowers102(args.root, transform=preprocess,
                              download=True, split="train")

      val_dataset = Flowers102(args.root, transform=preprocess,
                            download=True, split="val")
      
      test_dataset = Flowers102(args.root, transform=preprocess,
                            download=True, split="test")
      
      full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
      datasets_splits = train_val_test_dataset(full_dataset, test_split=0.3007, val_split=0.2851)
      
      train_dataset = datasets_splits["train"]
      val_dataset = datasets_splits["val"]
      test_dataset = datasets_splits["test"]

      assert len(train_dataset) == 4093 and len(val_dataset) == 1633 and len(test_dataset) == 2463
    
    elif args.dataset == "food101":
        train_dataset = Food101(args.root, transform=preprocess,
                              download=True, split="train")

        test_dataset = Food101(args.root, transform=preprocess,
                            download=True, split="test")
        
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        datasets_splits = train_val_test_dataset(full_dataset, test_split=0.3007, val_split=0.2851)
        train_dataset = datasets_splits["train"]
        val_dataset = datasets_splits["val"]
        test_dataset = datasets_splits["test"]

    elif args.dataset == "eurosat":
        dataset = EuroSAT(args.root, transform=preprocess, download=True)
        datasets_splits = train_val_test_dataset(dataset, test_split=0.5, val_split=0.4)
        
        train_dataset = datasets_splits["train"]
        val_dataset = datasets_splits["val"]
        test_dataset = datasets_splits["test"]

    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                        num_workers=args.num_workers, shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                        num_workers=args.num_workers, shuffle=False)

    
    if args.dataset == "cifar100" or args.dataset == "cifar10" or args.dataset == "food101":
        class_names = train_dataset.classes
    elif args.dataset == "flowers102":
        class_names = pd.read_csv("./oxford_flower_102_name.csv")["Name"]
        class_names = class_names.tolist()
    elif args.dataset == "eurosat":
        class_names = [
            "AnnualCrop", 
            "Forest", 
            "HerbaceousVegetation", 
            "Highway", 
            "Industrial", 
            "Pasture", 
            "PermanentCrops", 
            "Residential", 
            "River", 
            "SeaLake"
        ]

    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)
    
    with open(os.path.join(args.model_dir, "hyperparams.log"), "w") as fd:
        fd.write("\n".join(args_pretty_print))

    # wandb
    if args.use_wandb:
        wandb.init(project='vpt-exps-mit')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1 = validate(test_loader, texts, model, prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, texts, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break
    
    print("------- Testing ---------")
    loc = 'cuda:{}'.format(args.gpu)
    best_model_path = os.path.join(args.model_folder, "model_best.pth.tar")
    checkpoint = torch.load(best_model_path, map_location=loc)
    prompter.load_state_dict(checkpoint['state_dict'])

    validate(test_loader, texts, model, prompter, criterion, args, prefix="test")
    wandb.run.finish()


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                     })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


def validate(val_loader, texts, model, prompter, criterion, args, prefix="val"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progresmeter_prefix = "Validation" if prefix == "val" else "Test"
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='{}: '.format(progresmeter_prefix))

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt, _ = model(prompted_images, text_tokens)
            output_org, _ = model(images, text_tokens)
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1 = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(acc1[0].item(), images.size(0))

            acc1 = accuracy(output_org, target, topk=(1,))
            top1_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                f'{prefix}_loss': losses.avg,
                f'{prefix}_acc_prompt': top1_prompt.avg,
                f'{prefix}_acc_org': top1_org.avg,
            })

    return top1_prompt.avg


if __name__ == '__main__':
    wandb.login(key="087a5c3b07ae009d496e4c9369266f0f36fc0edc")
    gc.collect()
    torch.cuda.empty_cache()
    main()