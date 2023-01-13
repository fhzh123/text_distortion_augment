# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.model import TransformerModel, ClassifierModel
from model.dataset import CustomDataset
from utils import TqdmLoggingHandler, write_log
from task.utils import input_to_device

def augmenting(args):
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
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name, args.model_type)

    with h5py.File(os.path.join(save_path, 'processed.hdf5'), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        train_trg_list = f.get('train_label')[:]
        train_trg_list = F.one_hot(torch.tensor(train_trg_list, dtype=torch.long)).numpy()
        if args.model_type == 'bert':
            train_src_token_type_ids = f.get('train_src_token_type_ids')[:]
        else:
            train_src_token_type_ids = list()
        

    with open(os.path.join(save_path, 'word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        num_labels = data_['num_labels']
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    aug_model = TransformerModel(model_type=args.model_type, src_max_len=args.src_max_len,
                                 isPreTrain=args.isPreTrain, num_labels=num_labels,
                                 dropout=args.dropout, encoder_out_ratio=args.encoder_out_ratio)
    cls_model = ClassifierModel(d_latent=aug_model.d_hidden, num_labels=num_labels, dropout=args.dropout)
    aug_model.to(device)
    cls_model.to(device)

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                               src_seg_list=train_src_token_type_ids,
                               trg_list=train_trg_list, src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Model loading
    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss().to(device)
    save_file_name = os.path.join(args.model_save_path, args.data_name, args.model_type, 'checkpoint.pth.tar')
    checkpoint = torch.load(save_file_name)
    aug_model.load_state_dict(checkpoint['aug_model'])
    cls_model.load_state_dict(checkpoint['cls_model'])
    write_log(logger, f'Loaded augmenter model from {save_file_name}')

    #===================================#
    #=========Model Train Start=========#
    #===================================#
    
    start_time_e = time()
    write_log(logger, 'Augmenting start...')
    aug_model.eval()
    cls_model.eval()
    seq_list = list()
    aug_list = dict()
    for i in range(11):
        aug_list[f'eps_{i}'] = list()
        aug_list[f'eps_{i}_prob'] = list()

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        # Input setting
        b_iter = input_to_device(batch_iter, device=device)
        src_sequence, src_att, src_seg, trg_label = b_iter
        trg_label = torch.flip(trg_label, dims=[1])
        # trg_label = torch.full((len(trg_label), num_labels), 1 / num_labels).to(device)

        # Encoding
        with torch.no_grad():
            encoder_out = aug_model.encode(input_ids=src_sequence, attention_mask=src_att)
            latent_out = aug_model.latent_encode(input_ids=src_sequence, encoder_out=encoder_out)

        # Latent Encoding
        if aug_model.encoder_out_ratio == 0:
            latent_out_copy = latent_out.clone().detach().requires_grad_(True)
            hidden_states_copy = latent_out_copy
        elif aug_model.latent_out_ratio == 0:
            encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            hidden_states_copy = encoder_out_copy.max(dim=1)[0]
        else:
            encoder_out_copy = encoder_out.clone().detach().requires_grad_(True)
            latent_out_copy = latent_out.clone().detach().requires_grad_(True)
            hidden_states = \
            (aug_model.encoder_out_ratio * encoder_out_copy) + (aug_model.latent_out_ratio * latent_out_copy.unsqueeze(1))
            hidden_states_copy = hidden_states.clone().detach().requires_grad_(True)

        with torch.no_grad():
            recon_out = aug_model(input_ids=src_sequence, attention_mask=src_att, 
                                hidden_states=hidden_states_copy)
            origin_output = aug_model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)[0]

        classifier_out = aug_model.classify(hidden_states=hidden_states_copy)
        cls_loss = cls_criterion(classifier_out, trg_label)
        aug_model.zero_grad()
        cls_loss.backward()
        hidden_states_copy_grad = hidden_states_copy.grad.data

        src_output = aug_model.tokenizer.batch_decode(src_sequence, skip_special_tokens=True)
        seq_list.extend(src_output)

        for epsilon in range(11): # * 0.9
            hidden_states_copy = hidden_states_copy - ((epsilon * 1) * hidden_states_copy_grad)

            with torch.no_grad():
                recon_out = aug_model.generate(src_input_ids=src_sequence, src_attention_mask=src_att,
                                                latent_z=hidden_states_copy)

            # Augmenting
            aug_list[f'eps_{epsilon}'] = aug_model.tokenizer.batch_decode(recon_out.argmax(dim=2), skip_special_tokens=True)[0]

            with torch.no_grad():
                inp_dict = aug_model.tokenizer(aug_list[f'eps_{epsilon}'],
                                                max_length=args.src_max_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors='pt')
                encoder_out = aug_model.encode(input_ids=inp_dict['input_ids'].to(device), attention_mask=inp_dict['attention_mask'].to(device))
                latent_out = aug_model.latent_encode(input_ids=inp_dict['input_ids'].to(device), encoder_out=encoder_out)

                # Latent Encoding
                if aug_model.encoder_out_ratio == 0:
                    latent_out_copy = latent_out.clone().detach()
                    hidden_states_copy = latent_out_copy
                elif aug_model.latent_out_ratio == 0:
                    encoder_out_copy = encoder_out.clone().detach()
                    hidden_states_copy = encoder_out_copy.max(dim=1)[0]
                else:
                    encoder_out_copy = encoder_out.clone().detach()
                    latent_out_copy = latent_out.clone().detach()
                    hidden_states = \
                    (aug_model.encoder_out_ratio * encoder_out_copy) + (aug_model.latent_out_ratio * latent_out_copy.unsqueeze(1))
                    hidden_states_copy = hidden_states.clone().detach()

                classifier_out = aug_model.classify(hidden_states=hidden_states_copy)
                aug_list[f'eps_{epsilon}_prob'].extend(F.softmax(classifier_out).tolist())

        if args.debuging_mode:
            break
            
    result_dat = pd.DataFrame({
        'seq': seq_list,
        'aug_0': aug_list['eps_0'],
        'aug_1': aug_list['eps_1'],
        'aug_2': aug_list['eps_2'],
        'aug_3': aug_list['eps_3'],
        'aug_4': aug_list['eps_4'],
        'aug_5': aug_list['eps_5'],
        'aug_6': aug_list['eps_6'],
        'aug_7': aug_list['eps_7'],
        'aug_8': aug_list['eps_8'],
        'aug_9': aug_list['eps_9'],
        'aug_10': aug_list['eps_10']
    })
    result_dat.to_csv('./augmenting_result.csv', index=False)

    #===================================#
    #=========Tokenizing=========#
    #===================================#
