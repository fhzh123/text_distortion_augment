# Import modules
import os
import time
import argparse
# Import custom modules
from task.augmenter_training import augmenter_training
from task.text_style_transfer import style_transfer_training
from task.text_style_augmenting import style_transfer_augmenting
from task.augmenting import augmenting
from task.training import training
from task.test_textattack import test_textattack
# from task.testing import testing
# Utils
from utils import str2bool, path_check, set_random_seed

def main(args):

    # Time setting
    total_start_time = time.time()

    # Seed setting
    set_random_seed(args.random_seed)

    # Path setting
    path_check(args)

    if args.augmenter_training:
        augmenter_training(args)

    if args.style_transfer:
        style_transfer_training(args)

    if args.style_transfer_augmenting:
        style_transfer_augmenting(args)

    if args.augmenting:
        augmenting(args)

    if args.training:
        training(args)

    if args.test_textattack:
        test_textattack(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    user_name = os.getlogin()
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--augmenter_training', action='store_true')
    parser.add_argument('--augmenting', action='store_true')
    parser.add_argument('--style_transfer', action='store_true')
    parser.add_argument('--style_transfer_augmenting', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--test_textattack', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debuging_mode', action='store_true')
    # Path setting
    parser.add_argument('--data_name', default='IMDB', type=str,
                        help='Data name; Default is IMDB')
    parser.add_argument('--preprocess_path', default=f'/HDD/{user_name}/preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/HDD/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--model_save_path', default=f'/HDD/{user_name}/model_checkpoint/acl_text_aug', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default=f'/HDD/{user_name}/results/acl_text_aug', type=str,
                        help='Results file path')
    # Preprocessing setting
    parser.add_argument('--valid_ratio', default=0.2, type=float,
                        help='Validation split ratio; Default is 0.2')
    parser.add_argument('--min_len', default=4, type=int,
                        help="Sentences's minimum length; Default is 4")
    parser.add_argument('--src_max_len', default=50, type=int,
                        help="Sentences's minimum length; Default is 50")
    # Augmenter setting
    parser.add_argument('--isPreTrain', default=True, type=str2bool,
                        help='Use pre-trained language model; Default is True')
    parser.add_argument('--aug_encoder_model_type', default='T5', type=str,
                        help='Classification model type; Default is T5')
    parser.add_argument('--aug_decoder_model_type', default='T5', type=str,
                        help='Augmentation model type; Default is T5')
    parser.add_argument('--classify_method', default='encoder_out', type=str, choices=['encoder_out', 'latent_out'],
                        help='Classification method; Default is encoder_out')
    parser.add_argument('--encoder_out_mix_ratio', default=1, type=float,
                        help='Encoder output ratio to input of decoder; Default is 1')
    parser.add_argument('--encoder_out_cross_attention', default=True, type=str2bool,
                        help='Use cross attention of decoder via encoder out; Default is True')
    parser.add_argument('--encoder_out_to_augmenter', default=True, type=str2bool,
                        help='Add decoder output and encoder output to augmenter; Default is True')
    parser.add_argument('--latent_out_to_augmenter', default=True, type=str2bool,
                        help='Add decoder output and latent output to augmenter; Default is True')
    parser.add_argument('--latent_mmd_loss', default=False, type=str2bool,
                        help='Latent variable MMD loss; Default is False')
    parser.add_argument('--label_flipping', default=False, type=str2bool,
                        help='Label flipping; Default is False')
    # Classifier setting
    parser.add_argument('--cls_model_type', default='albert', type=str,
                        help='Classification model type; Default is ALBERT')
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--aug_cls_optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'Ralamb', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--aug_cls_scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--aug_recon_optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'Ralamb', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--aug_recon_scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--cls_optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'Ralamb', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--cls_scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--aug_cls_lr', default=5e-4, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-4')
    parser.add_argument('--aug_recon_lr', default=5e-4, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-4')
    parser.add_argument('--cls_lr', default=5e-4, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-4')
    parser.add_argument('--n_warmup_epochs', default=2, type=float,
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--train_with_aug', action='store_true')
    parser.add_argument('--aug_cls_num_epochs', default=5, type=int,
                        help='Classifier training epochs; Default is 5')
    parser.add_argument('--aug_recon_num_epochs', default=10, type=int,
                        help='Augmenter training epochs; Default is 10')
    parser.add_argument('--training_num_epochs', default=10, type=int,
                        help='Classifier training epochs; Default is 10')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size; Default is 16')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int,
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--recon_label_smoothing_eps', default=0.05, type=float,
                        help='Label smoothing epsilon; Default is 0.05')
    parser.add_argument('--cls_label_smoothing_eps', default=0.05, type=float,
                        help='Label smoothing epsilon; Default is 0.05')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout ratio; Default is 0.3')
    parser.add_argument('--z_variation', default=2, type=float,
                        help='')
    # Testing setting
    parser.add_argument('--grad_epsilon', default=0.1, type=float,
                        help='')
    parser.add_argument('--epsilon_repeat', default=3, type=int,
                        help='')
    parser.add_argument('--test_batch_size', default=32, type=int,
                        help='Test batch size; Default is 32')
    parser.add_argument('--test_decoding_strategy', default='beam', choices=['greedy', 'beam', 'multinomial', 'topk', 'topp', 'midk'], type=str,
                        help='Decoding strategy for test; Default is beam')
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam search size; Default is 5')
    parser.add_argument('--beam_alpha', default=0.7, type=float,
                        help='Beam search length normalization; Default is 0.7')
    parser.add_argument('--repetition_penalty', default=1.3, type=float,
                        help='Beam search repetition penalty term; Default is 1.3')
    parser.add_argument('--topk', default=5, type=int,
                        help='Topk sampling size; Default is 5')
    parser.add_argument('--topp', default=0.9, type=float,
                        help='Topk sampling size; Default is 0.9')
    parser.add_argument('--midk', default=2, type=int,
                        help='Midk sampling size; Default is 2; Refer to model.py')
    parser.add_argument('--multinomial_temperature', default=1.0, type=float,
                        help='Multinomial sampling temperature; Default is 1.0')
    parser.add_argument('--augmenting_label', default='hard', type=str,
                        help='')
    # augmentation setting
    parser.add_argument('--augmenting_target', default='both', type=str,
                        help='it only works for rte whtch is the only dataset that has two targets')
    # Seed & Logging setting
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed; Default is 42')
    parser.add_argument('--print_freq', default=300, type=int,
                        help='Print training process frequency; Default is 300')
    parser.add_argument('--print_example', default=True, type=str2bool,
                        help='Print augmented example; Default is True')
    parser.add_argument('--sampling_ratio', default=0.1, type=float,
                        help='')
    args = parser.parse_args()

    main(args)
