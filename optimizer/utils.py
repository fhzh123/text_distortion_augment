from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from .optimizer import Ralamb
from .scheduler import WarmupLinearSchedule

from transformers import AdamW

def optimizer_select(optimizer_model, model, lr, w_decay):

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-5
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    if optimizer_model == 'SGD':
        optimizer = optim.SGD(optimizer_grouped_parameters, lr, momentum=0.9, weight_decay=w_decay)
    elif optimizer_model == 'Adam':
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr, eps=1e-8, weight_decay=w_decay)
    elif optimizer_model == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, weight_decay=w_decay)
    elif optimizer_model == 'Ralamb':
        optimizer = Ralamb(optimizer_grouped_parameters, lr=lr, weight_decay=w_decay)
    else:
        raise Exception("Choose optimizer in ['AdamW', 'Adam', 'SGD', 'Ralamb']")
    return optimizer

def shceduler_select(phase, scheduler_model, optimizer, dataloader_len, args):

    if phase == 'cls':
        num_epochs = args.aug_cls_num_epochs
    elif phase == 'aug':
        num_epochs = args.aug_recon_num_epochs
    elif phase == 'training':
        num_epochs = args.training_num_epochs

    # Scheduler setting
    if scheduler_model == 'constant':
        scheduler = StepLR(optimizer, step_size=dataloader_len, gamma=1)
    elif scheduler_model == 'warmup':
        scheduler = WarmupLinearSchedule(optimizer,
                                        warmup_steps=int(dataloader_len*args.n_warmup_epochs),
                                        t_total=dataloader_len*num_epochs)
    elif scheduler_model == 'reduce_train':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(dataloader_len*1.5),
                                      factor=0.5)
    elif scheduler_model == 'reduce_valid':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    elif scheduler_model == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_lambda ** epoch)
    else:
        raise Exception("Choose shceduler in ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']")
    return scheduler