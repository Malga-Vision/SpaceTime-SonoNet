"""
sononet2d-traintest.py:    This file trains the 2D SonoNet model on US dataset and runs test inference.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import json
import torch
import pickle
import argparse
import glob
from torch.utils.data import DataLoader
from sononet2d.models import SonoNet2D
from utils.datareader import USDataset2D_comparable_3d
from utils.datasplit import split3d_train_validation, split_videos_in_clips
from utils.runner import NoGPUError, train, test
from utils.visualize import plot_confusion, plot_history
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report




IMG_SIZE = [256, 256]
NUM_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-pretrain_dir', type=str, default=None, help='Directory of pre-training results.')
    parser.add_argument('-gpu', nargs='*', type=int, default=[0],
                        help='Choose the device number of your GPU. Default is 0.')
    parser.add_argument('-num_features', type=int, default=16,
                        help='Number of hidden features for the SonoNet model: either 16, 32 or 64. Default is 16.')
    parser.add_argument('-validation_split', type=float, default=0.2)
    parser.add_argument('-batch_size', type=int, default=5)
    parser.add_argument('-patience', type=int, default=6)
    parser.add_argument('-max_num_epochs', type=int, default=50)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-lr_sched_patience', type=int, default=5)
    parser.add_argument('-clip_len', type=int, default=5, help='Number of frame in a clip. Default is 5.')
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-blocks_to_freez', type=int, default=0, help='Number of layer to freeze for fine-tuning')
    parser.add_argument('--train_classifier_only', action="store_true",
                       help='Specify this if you want to train only the head of the network instead than all layers.')
    parser.add_argument('--sampler', action="store_true", help = 'Specify this if you want use a random weighted sampler to have a balance batch distribution.')
    parser.add_argument('--augmentation', action="store_true", help = 'Specify this if you want use data augmentation.')
    return parser.parse_args()




def main():

    # Load all given parameters --------------------------------------------------------------------#
    args = parse_args()
    log_dir = os.path.join(args.log_dir, f'SonoNet-{args.num_features}')
    sub_dir = str(args.lr) + "-" + str(args.batch_size) +"-"  + args.optimizer + '_' + str(args.seed)
    log_dir = os.path.join(log_dir, sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pretrain_dir = None
    if args.pretrain_dir:
        pretrain_dir = os.path.join(args.pretrain_dir, f'SonoNet-{args.num_features}')

    # Check if GPU is available --------------------------------------------------------------------#
    if not any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        raise NoGPUError('No torch-compatible GPU found. Aborting.')
    multigpu = False
    if torch.cuda.is_available():
        if len(args.gpu) > 1:
            multigpu = True
            device = torch.device('cuda')
            all_devices = [torch.device(f'cuda:{i}') for i in args.gpu]
            print(f'\nWill use CUDA-compatible GPU number {args.gpu}:')
            for dev in all_devices:
                print(f'     - {torch.cuda.get_device_name(dev)}')
        else:
            device = torch.device(f'cuda:{args.gpu[0]}')
            print(f'\nWill use CUDA-compatible GPU number {args.gpu[0]}: {torch.cuda.get_device_name(device)}')
    else:  # torch.backends.mps.is_available()
        device = torch.device("mps")
        print('\nWill use MPS-compatible GPU')
    
    # Load training dataset ------------------------------------------------------------------------#
    print('\nLoading training data...')
    
    print("seed for split: ", args.seed)
    train_clips, val_clips = split3d_train_validation(args.data_dir, valid_split=args.validation_split, clip_size=args.clip_len, verbose = True, random_seed=args.seed)
    print(train_clips[0])
    
    train_paths = [p[int(args.clip_len / 2)] for p in train_clips]
    val_paths = [p[int(args.clip_len / 2)] for p in val_clips]
    
    train_dataset = USDataset2D_comparable_3d(root=args.data_dir, img_paths=train_paths, split="train",
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS, augmentation=args.augmentation)
    val_dataset = USDataset2D_comparable_3d(root=args.data_dir, img_paths=val_paths, split="val",
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS)

    class_counts = train_dataset.samples_per_class
    print("Sample per class: ", class_counts)
    # Random Sampler -----------------------------------------------------------------------------------#
    if args.sampler:
        class_weights = 1. / class_counts
        if not os.path.exists(os.path.join(args.data_dir, f"sampler_weight_{args.seed}.npy")):
            print("Computing sampler weights.......")
            y_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
            samples_weight = np.array([class_weights[t] for t in y_labels])
            samples_weight = torch.from_numpy(samples_weight)
            np.save(os.path.join(args.data_dir, f"sampler_weight_{args.seed}.npy"), samples_weight)
        else:
            samples_weight = np.load(os.path.join(args.data_dir, f"sampler_weight_{args.seed}.npy"))
        print("Sampler weights: ", samples_weight)
    
        sampler = WeightedRandomSampler(weights = samples_weight, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(dataset=train_dataset, sampler = sampler, batch_size=args.batch_size, pin_memory=True, num_workers=4)
        train_class_weights = None   #quando uso sampler non uso weighted losses
    
    
    if args.sampler == False:
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=4)
        train_class_weights = None

    
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    
    
    # Build the model ------------------------------------------------------------------------------#
    print('\nBuilding network...')
    net = SonoNet2D(hid_features=args.num_features, out_labels=train_dataset.num_classes, train_classifier_only=args.train_classifier_only)
    if multigpu:
        net = torch.nn.DataParallel(net, device_ids=args.gpu)
    net.to(device)
    del train_dataset, val_dataset
    # Training -------------------------------------------------------------------------------------#
    print('\nStarting training phase...')
    _, history = train(net, train_loader, val_loader,
                       pretraining_dir=pretrain_dir,
                       checkpoint_dir=log_dir, optimizer=args.optimizer, 
                       class_weights=train_class_weights, 
                       max_num_epochs=args.max_num_epochs, patience=args.patience,
                       lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                       lr_scheduler='plateau', lr_sched_patience=args.lr_sched_patience,
                       device=device, seed=args.seed)
    
    # Load testing dataset -------------------------------------------------------------------------#
    print('\nLoading test data...')
    test_dirs =  sorted(glob.glob(os.path.join(args.data_dir, 'test', 'videos', '*')))
    test_paths = list(map(lambda p: sorted(glob.glob(os.path.join(p, '*.png'))), test_dirs))
    test_clips = split_videos_in_clips(test_paths, args.data_dir, clip_size=args.clip_len)
    test_paths = [p[int(args.clip_len / 2)] for p in test_clips]
    
    test_dataset = USDataset2D_comparable_3d(root=args.data_dir, img_paths=test_paths, split="test",
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)


    # Testing --------------------------------------------------------------------------------------#
    print('\nStarting test phase...')
    [test_f1, test_acc], [predictions, targets] = test(net, test_loader,
                                                        checkpoint_dir=log_dir,
                                                        device=device)
    
    history = pickle.load(open(os.path.join(log_dir, "history"), "rb"))
    val_loss = min(history['val_loss']) 
    val_loss_index = history['val_loss'].index(val_loss)
    val_acc = history['val_accuracy'][val_loss_index]
    val_f1 = history['val_f1score'][val_loss_index]

    
    # Store the test results
    results = { 'accuracy': round(test_acc * 100, 3), 'f1score': round(test_f1, 3), 'val accuracy': round(val_acc * 100, 3), 'val loss': round(val_loss, 3), 'val f1': round(val_f1, 3)}
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    class_report_dict = classification_report(torch.Tensor(targets), torch.Tensor(predictions), target_names=target_names, output_dict=True)
    eval_list = [results, class_report_dict ] 
    eval_file = os.path.join(log_dir, 'evaluation.json')    
    with open(eval_file, "w+") as f:
        json.dump(eval_list, f, indent=2)

    targ_file = os.path.join(log_dir, 'targets.pickle')
    with open(targ_file, 'wb') as f:
        pickle.dump(targets, f)
    pred_file = os.path.join(log_dir, 'predictions.pickle')
    with open(pred_file, 'wb') as f:
        pickle.dump(predictions, f)
    print('\nDONE.')

    plot_history(os.path.join(log_dir, "history"), log_dir=log_dir, format='png')
    plot_confusion(predictions, targets, num_classes=test_dataset.num_classes, log_dir=log_dir, format='png')
    plot_history(os.path.join(log_dir, "history"), log_dir=log_dir, format='svg')
    plot_confusion(predictions, targets, num_classes=test_dataset.num_classes, log_dir=log_dir, format='svg')    
    del test_dataset, predictions, targets
if __name__ == '__main__':
    main()
