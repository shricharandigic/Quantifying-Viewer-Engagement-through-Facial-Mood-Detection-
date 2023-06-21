"""
Based on
Farzaneh_2021_WACV,
    author    = {Farzaneh, Amir Hossein and Qi, Xiaojun},
    title     = {Facial Expression Recognition in the Wild via Deep Attentive Center Loss},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2402-2411}
    reference = https://github.com/amirhfarzaneh/dacl
"""

import random
import pprint
import time
import sys
import os
import socket

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Logger, AverageMeter, accuracy, calc_metrics, RandomFiveCrop
from datetime import datetime, timedelta
from loss import SparseCenterLoss
from resnet import resnet18



def main(cfg):
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if cfg['deterministic']:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Loading RAF-DB
    # -----------------
    print('[>] Loading dataset '.ljust(64, '-'))
    normalize = transforms.Normalize(mean=[0.5752, 0.4495, 0.4012],
                                     std=[0.2086, 0.1911, 0.1827])
    # Loading train set
    train_set = datasets.ImageFolder(
        root=os.path.join(cfg['root_dir'], 'train'),
        transform=transforms.Compose([
            transforms.Resize(256),
            RandomFiveCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['workers'], pin_memory=True)

    # Loading validation set
    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(cfg['root_dir'], 'test'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['workers'], pin_memory=True
    )
    print('[*] Loaded dataset!')

    # Creating Model
    # ------------
    print('[>] Model '.ljust(64, '-'))
    if cfg['arch'] == 'resnet18':
        feat_size = 512
        if not cfg['pretrained'] == '':
            model = resnet18(pretrained=cfg['pretrained'])
            model.fc = nn.Linear(feat_size, 7)
        else:
            print('[!] model is trained from scratch!')
            model = resnet18(num_classes=7, pretrained=cfg['pretrained'])
    else:
        raise NotImplementedError('only working with "resnet18" now! check cfg["arch"]')
    model = torch.nn.DataParallel(model).to(device)
    print('[*] Model initialized!')

    # Defining loss function (criterion) and optimizer
    # ----------------------------------------------
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
        'center': SparseCenterLoss(7, feat_size).to(device)
    }
    optimizer = {
        'softmax': torch.optim.SGD(model.parameters(), cfg['lr'],
                                   momentum=cfg['momentum'],
                                   weight_decay=cfg['weight_decay']),
        'center': torch.optim.SGD(criterion['center'].parameters(), cfg['alpha'])
    }
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer['softmax'], step_size=20, gamma=0.1)

    # Starting training and evaluation

    global best_validation
    best_validation = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)

    print('[>] Begin Training '.ljust(64, '-'))
    for epoch in range(1, cfg['epochs'] + 1):
        start = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        # validate for one epoch
        validate(val_loader, model, criterion, epoch, cfg)

        # progress
        end = time.time()
        progress = (
            f'[*] epoch time = {end - start:.2f}s | '
            f'lr = {optimizer["softmax"].param_groups[0]["lr"]}\n'
        )
        print(progress)

        # lr step
        scheduler.step()

    torch.save(model.state_dict(), "./model_state/m1.pt")
    torch.save(model, "./model/m1.pt")

    # best validation info
    # ---------------
    print('[>] best validation '.ljust(64, '-'))
    stat = (
        f'[+] acc={best_validation["acc"]:.4f}\n'
        f'[+] rec={best_validation["rec"]:.4f}\n'
        f'[+] f1={best_validation["f1"]:.4f}\n'
        f'[+] aucpr={best_validation["aucpr"]:.4f}\n'
        f'[+] aucroc={best_validation["aucroc"]:.4f}'
    )
    print(stat)


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to train mode
    model.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            feat, output, A = model(images)
            l_softmax = criterion['softmax'](output, target)
            l_center = criterion['center'](feat, A, target)
            l_total = l_softmax + cfg['lamb'] * l_center

            # measure accuracy and record loss
            acc, pred = accuracy(output, target)
            losses['softmax'].update(l_softmax.item(), images.size(0))
            losses['center'].update(l_center.item(), images.size(0))
            losses['total'].update(l_total.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(output.data)

            # compute grads + opt step
            optimizer['softmax'].zero_grad()
            optimizer['center'].zero_grad()
            l_total.backward()
            optimizer['softmax'].step()
            optimizer['center'].step()

            # progressbar
            pbar.set_description(f'TRAINING [{epoch:03d}/{cfg["epochs"]}]')
            pbar.set_postfix({'L': losses["total"].avg,
                              'Ls': losses["softmax"].avg,
                              'Lsc': losses["center"].avg,
                              'acc': accs.avg})
            pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] TRAIN [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)
    write_log(losses, accs.avg, metrics, epoch, tag='train')


def validate(valid_loader, model, criterion, epoch, cfg):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to evaluate mode
    model.eval()

    with tqdm(total=int(len(valid_loader.dataset) / cfg['batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):
                images = images.to(device)
                target = target.to(device)

                # compute output
                feat, output, A = model(images)
                l_softmax = criterion['softmax'](output, target)
                l_center = criterion['center'](feat, A, target)
                l_total = l_softmax + cfg['lamb'] * l_center

                # measure accuracy and record loss
                acc, pred = accuracy(output, target)
                losses['softmax'].update(l_softmax.item(), images.size(0))
                losses['center'].update(l_center.item(), images.size(0))
                losses['total'].update(l_total.item(), images.size(0))
                accs.update(acc.item(), images.size(0))

                # collect for metrics
                y_pred.append(pred)
                y_true.append(target)
                y_scores.append(output.data)

                # progressbar
                pbar.set_description(f'VALIDATING [{epoch:03d}/{cfg["epochs"]}]')
                pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] VALID [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)

    # save model checkpoints for best validation
    if accs.avg > best_validation['acc']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_acc.pth')
    if metrics['rec'] > best_validation['rec']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_rec.pth')

    best_validation['acc'] = max(best_validation['acc'], accs.avg)
    best_validation['rec'] = max(best_validation['rec'], metrics['rec'])
    best_validation['f1'] = max(best_validation['f1'], metrics['f1'])
    best_validation['aucpr'] = max(best_validation['aucpr'], metrics['aucpr'])
    best_validation['aucroc'] = max(best_validation['aucroc'], metrics['aucroc'])
    write_log(losses, accs.avg, metrics, epoch, tag='valid')


def write_log(losses, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'L_softmax/{tag}', losses['softmax'].avg, e)
    writer.add_scalar(f'L_center/{tag}', losses['center'].avg, e)
    writer.add_scalar(f'L_total/{tag}', losses['total'].avg, e)
    writer.add_scalar(f'acc/{tag}', acc, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)


def save_checkpoint(epoch, model, cfg, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(cfg['save_path'], tag))


if __name__ == '__main__':
    cfg = {
        # global config
        'log_dir': "logs",
        'root_dir': "./path/to/fer/dataset",
        'deterministic': False,
        'seed': 0,
        'workers': 4,
        # training config
        'arch': 'resnet18',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 64,
        'epochs': 25,
        'pretrained': "imagenet",
        # centerloss config
        'alpha': 0.5,
        'lamb': 0.01
    }
    tag = (
        f'_a-centerloss'
        f'_ar={cfg["arch"]}'
        f'_pt={cfg["pretrained"]}'
        f'_bs={cfg["batch_size"]}'
        f'_lr={cfg["lr"]}'
        f'_wd={cfg["weight_decay"]}'
        f'_alpha={cfg["alpha"]}'
        f'_lamb={cfg["lamb"]}'
    )

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logname = current_time + '_' + socket.gethostname() + tag
    cfg['save_path'] = os.path.join(cfg['log_dir'], logname)

    # setting up writers
    global writer
    writer = SummaryWriter(cfg['save_path'])
    sys.stdout = Logger(os.path.join(cfg['save_path'], 'log.log'))

    # print finalized parameter config
    print('[>] Configuration '.ljust(64, '-'))
    pp = pprint.PrettyPrinter(indent=2)
    print(pp.pformat(cfg))

    # -----------------
    start = time.time()
    main(cfg)
    end = time.time()
    # -----------------

    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
