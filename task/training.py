# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import logging
import datetime
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import custom modules
from model.dataset import CustomDataset
from model.loss import compute_mmd, CustomLoss
from optimizer.utils import shceduler_select, optimizer_select
from optimizer.scheduler import get_cosine_schedule_with_warmup
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import data_load, data_sampling, aug_data_load, tokenizing, input_to_device, encoder_parameter_grad

def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    start_time = time()
    total_src_list_, total_trg_list_ = data_load(args)
    total_src_list, total_trg_list = data_sampling(args, total_src_list_, total_trg_list_)
    num_labels = len(set(total_trg_list['train']))

    # One-hot encoding
    total_trg_list['train'] = F.one_hot(torch.tensor(total_trg_list['train'], dtype=torch.long)).numpy()
    total_trg_list['valid'] = F.one_hot(torch.tensor(total_trg_list['valid'], dtype=torch.long)).numpy()
    total_trg_list['test'] = F.one_hot(torch.tensor(total_trg_list['test'], dtype=torch.long)).numpy()

    if args.train_with_aug:
        aug_src_list, aug_trg_list = aug_data_load(args)
        total_src_list['train'] = np.append(total_src_list['train'], aug_src_list)
        total_trg_list['train'] = np.concatenate((total_trg_list['train'], aug_trg_list), axis=0)

    write_log(logger, 'Data loading done!')

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)
    model.to(device)

    # 2) Dataloader setting
    # tokenizer_name = return_model_name(args.encoder_model_type)
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    src_vocab_num = tokenizer.vocab_size

    dataset_dict = {
        'train': CustomDataset(tokenizer=tokenizer,
                               src_list=total_src_list['train'], src_list2=total_src_list['train2'],
                               trg_list=total_trg_list['train'], src_max_len=args.src_max_len),
        'valid': CustomDataset(tokenizer=tokenizer,
                               src_list=total_src_list['test'], src_list2=total_src_list['test'],
                               trg_list=total_trg_list['test'], src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(optimizer_model=args.training_optimizer, model=model, lr=args.aug_lr, w_decay=args.w_decay)
    scheduler = shceduler_select(phase='training', scheduler_model=args.training_scheduler, optimizer=optimizer, dataloader_len=len(dataloader_dict['train']), args=args)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(label_smoothing=args.training_label_smoothing_eps).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_path = os.path.join(args.model_save_path, args.data_name)
        save_file_name = os.path.join(save_path,
                                        f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+4

    for epoch in range(start_epoch + 1, args.aug_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Training start...')
        model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            optimizer.zero_grad(set_to_none=True)

            # Input setting
            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            #===================================#
            #========Augmenter Training=========#
            #===================================#

            # Augmenter training
            logit = model(input_ids=src_sequence, attention_mask=src_att)['logits']

            loss = criterion(logit, trg_label)
            loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            # Accuracy
            acc = sum((logit.argmax(dim=1) == trg_label.argmax(dim=1))) / len(trg_label)

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_acc:%03.2f | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, loss.item(), acc.item(), optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

        #===================================#
        #============Validation=============#
        #===================================#
        write_log(logger, 'Validation start...')

        # Validation
        model.eval()

        val_loss = 0
        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            b_iter = input_to_device(batch_iter, device=device)
            src_sequence, src_att, src_seg, trg_label = b_iter

            with torch.no_grad():
                # Augmenter training
                logit = model(input_ids=src_sequence, attention_mask=src_att)['logits']
                loss = criterion(logit, trg_label)

                val_loss += loss
                val_acc += sum((logit.argmax(dim=1) == trg_label.argmax(dim=1))) / len(trg_label)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Augmenter Validation CrossEntropy Loss: %3.3f' % val_recon_loss)
        write_log(logger, 'Augmenter Validation MMD Loss: %3.3f' % val_mmd_loss)

        save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, f'cls_checkpoint_seed_{args.random_seed}.pth.tar')
        if val_recon_loss < best_aug_val_loss:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_file_name)
            best_aug_val_loss = val_recon_loss
            best_aug_epoch = epoch
        else:
            else_log = f'Still {best_aug_epoch} epoch Loss({round(best_aug_val_loss.item(), 2)}) is better...'
            write_log(logger, else_log)

    # 3) Results
    write_log(logger, f'Best AUG Epoch: {best_aug_epoch}')
    write_log(logger, f'Best AUG Loss: {round(best_aug_val_loss.item(), 2)}')
    write_log(logger, f'Best CLS Epoch: {best_cls_epoch}')
    write_log(logger, f'Best CLS Loss: {round(best_cls_val_loss.item(), 2)}')