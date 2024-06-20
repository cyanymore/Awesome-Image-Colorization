import argparse
import random
import numpy as np
import torch
from src.metrics import evaluation
import src.train as train
import train_kaist
from src.utils import set_deterministic
from src.sample import sample


def check_errors(args):
    if args.real and args.qsn and args.phm:
        raise Exception("Check or real or qsn or phm, not all togheter")
    if args.real and args.qsn:
        print(args.real, args.qsn)
        raise Exception("Check or real or qsn, not all togheter")
    if args.real and args.phm:
        raise Exception("Check or real or phm, not all togheter")
    if args.phm and args.qsn:
        raise Exception("Check or qsn or phm, not all togheter")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_bool = map(str2bool, strings.split(","))
    return tuple(mapped_bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-dataset', type=str, default='droneveichle')
    parser.add_argument('-dataset_path', type=str, default='dataset')
    parser.add_argument('-sample_dir', type=str, default='sample')
    parser.add_argument('-experiment_name', type=str, default='testing')
    parser.add_argument('-eval_dir', type=str, default='results')
    parser.add_argument('-save_path', type=str, default='checkpoints')
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-logs_every', type=int, default=10)
    parser.add_argument('-eval_every', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=12)
    parser.add_argument('-eval_batch_size', type=int, default=50)
    parser.add_argument('-img_size', type=int, default=256)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-sepoch', type=int, default=0)
    parser.add_argument("-preloaded_data", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument('-tensors_path', type=str, default='/tensors/tensors_nodark')
    parser.add_argument("-preloaded_data_eval", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-pretrained_generator", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-color_images", type=str2bool, nargs='?', const=True, default=True, help="Activate nice mode.")
    parser.add_argument("-lab", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-classes", type=tuple_type, default=(False, False)) #first if injected second if style extracted
    parser.add_argument("-classes_image", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-single_mod", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.") 
    parser.add_argument("-remove_dark_samples", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.") 
    parser.add_argument('-wavelet_type', type=str,default=None) #real or quat
    parser.add_argument("-loss_ssim", type=str2bool, nargs='?', const=True, default=True, help="Activate nice mode.")
    parser.add_argument('-w_ssim', type=float, default=1)
    parser.add_argument("-tv_loss", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument('-w_tv', type=float, default=5)
    parser.add_argument("-spectral", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-groupnorm", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-contrast_t", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument('-weight_l1', type=float, default=1)
    parser.add_argument("-alternate_target", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-decay", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")

    parser.add_argument("-real", type=str2bool, nargs='?', const=True, default=True, help="Activate nice mode.")
    parser.add_argument("-qsn", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")
    parser.add_argument("-phm", type=str2bool, nargs='?', const=True, default=False, help="Activate nice mode.")


    parser.add_argument('-gan_version', type=str, default='Generator[2/3]+shapeunet+D')
    parser.add_argument('-modals', type=tuple, default=("img_ir", "img"))
    parser.add_argument('-lr', type=float, default=2e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-random_seed', type=int, default='888')
    parser.add_argument('-c_dim', type=int, default='2')
    parser.add_argument('-h_conv', type=int, default='16')
    parser.add_argument('-G_conv', type=int, default='64')
    parser.add_argument('-layers_gen', type=int, default='2')
    parser.add_argument('-s_layers_gen', type=int, default='3')
    parser.add_argument('-r_lay', type=int, default='256')


    parser.add_argument('-betas', type=tuple, default=(0.5, 0.9))
    parser.add_argument('-ttur', type=float, default=3e-4)
    parser.add_argument('-w_d_false_c', type=float, default=0.01)
    parser.add_argument('-w_d_false_t_c', type=float, default=0.01)
    parser.add_argument('-w_g_c', type=float, default=1.0)
    parser.add_argument('-w_g_t_c', type=float, default=1.0)
    parser.add_argument('-w_g_cross', type=float, default=25)
    parser.add_argument('-w_shape', type=float, default=1)
    parser.add_argument('-w_cycle', type=float, default=10)
    
    args = parser.parse_args()

    check_errors(args)
    print(args)
    set_deterministic(args.random_seed)
    if "train" in args.mode:
        if "debug" in args.mode:
            # args.img_size = 128
            args.preloaded_data = False
        if args.dataset=="droneveichle":
            train.train(args)
        else:
            train_kaist.train(args)
    if "eval" in args.mode:
        if "debug" in args.mode:
            args.eval_batch_size = 1
        evaluation(args)
    if "sample" in args.mode:
        if "debug" in args.mode:
            args.eval_batch_size = 1
        sample(args)