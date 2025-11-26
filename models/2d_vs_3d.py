from sononet2d.models import SonoNet2D
from sononet3d.models import SonoNet3D, SonoNet3D_2_1d
from utils.datareader import USDataset2D
import os
import argparse
import torch
from utils.runner import NoGPUError, test, test_3d
import glob
from torch.utils.data import Dataset
from typing import List
from torchvision import transforms
import json
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from datetime import datetime
from collections import Counter


IMG_SIZE = [256, 256]
NUM_CHANNELS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-model_dir_2d', type=str, help='Directory of the 2d model to load.')
    parser.add_argument('-model_dir_3d', type=str, help='Directory of the 3d model to load.')
    parser.add_argument('-model_dir_2d1d', type=str, help='Directory of the 2d1d model to load.')
    parser.add_argument('-num_features', type=int, default=32,
                        help='Number of hidden features for the SonoNet model: either 16, 32 or 64. Default is 16.')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpu', nargs='*', type=int, default=[0], help='Choose the device number of your GPU. Default is 0.')
    parser.add_argument('-clip_len', type=int, default=10, help='Number of frame in a clip. Default is 10.')
    return parser.parse_args()


def split_video_in_clips(frame_path, clip_len):
    clips = [frame_path[i:i+clip_len]  for i in range(len(frame_path)) if len(frame_path[i:i+clip_len]) == clip_len]

    return clips



def compute_accuracy_per_class(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError("Length of targets and predictions must be the same")

    classes = set(targets)
    total_counts = Counter(targets)
    
    correct_counts = Counter()
    for target, prediction in zip(targets, predictions):
        if target == prediction:
            correct_counts[target] += 1
    
    accuracy_per_class = {}
    for cls in classes:
        accuracy_per_class[cls] = correct_counts[cls] / total_counts[cls]
    
    return accuracy_per_class


def count_samples_per_class(clips, classes):
    target_classes = list(classes.values())
    class_counts_dict = {class_num: 0 for class_num in target_classes}
    for clip in clips:
        with open(clip[-1].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:
            lbl = int(classes[f.read()])
        class_counts_dict[lbl] += 1
    return list(class_counts_dict.values())


class Normalize:
    """Perform normalization and fill NaNs with zeros."""

    def __init__(self):
        pass

    def __call__(self, image):
        lastdim = len(image.size()) - 1
        image /= torch.amax(image, dim=(lastdim-1, lastdim))[..., None, None]
        return torch.where(torch.isnan(image), torch.zeros_like(image), image)
    

class USDataset3D(Dataset):
    def __init__(self, root: str, clips_paths, in_channels: int = 1, img_size: List[int] = [256, 256]):   
        
        self.clips_paths = clips_paths         
        self.num_channels = in_channels
        self.img_size = img_size
        with open(os.path.join(root, 'classes.json'), 'r') as f:
            self.classes = json.load(f)

        self.num_classes = len(self.classes)
        self.clip_len = len(self.clips_paths[0])
        self.num_clips = len(self.clips_paths)

        self.samples_per_class = np.array(count_samples_per_class(self.clips_paths, self.classes))

        self.basic_transforms = transforms.Compose(
            [transforms.PILToTensor(), transforms.Grayscale(num_output_channels=self.num_channels), transforms.ConvertImageDtype(dtype=torch.float), 
             transforms.Resize(img_size, antialias=True) , Normalize()] 
        )

    def __getitem__(self, index):
        clip_path = self.clips_paths[index]
        clip_frames = torch.zeros([self.clip_len, self.num_channels, *self.img_size], dtype=torch.float)

        for idx, frame_path in enumerate(clip_path):
            clip_frames[idx] = self.basic_transforms(default_loader(frame_path))
        
        with open(clip_path[-1].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:
            lbl = int(self.classes[f.read()])

        return clip_frames, lbl
    
    def __len__(self):
        return self.num_clips


def main():

    args = parse_args()

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

    
    start_frame_index = args.clip_len - 1
    with open(os.path.join(args.data_dir, 'classes.json'), 'r') as f:
        classes = json.load(f)
    num_classes = len(list(classes.values()))
    print("num classes: ", num_classes)

    
    net_2d = SonoNet2D(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=num_classes, train_classifier_only=False)
    
    net_3d = SonoNet3D(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=num_classes, train_classifier_only=False)
    
    net_2d1d = SonoNet3D_2_1d(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=num_classes, train_classifier_only=False)
    
   
    net_2d.to(device)
    path_weights_2d =  args.model_dir_2d
    net_2d1d.to(device)
    path_weights_2d1d =  args.model_dir_2d1d

    net_3d.to(device)
    path_weights_3d =  args.model_dir_3d


    test_video_dir = os.listdir(os.path.join(args.data_dir, 'test/videos'))
    print("Number of video in test set: ", len(test_video_dir))

    accuracy_2d = [] 
    accuracy_3d = []
    accuracy_2d1d = []
    pred_2d_list = []
    target_2d_list = []
    pred_3d_list = []
    target_3d_list = []
    pred_2d1d_list = []
    target_2d1d_list = []

    pbar = tqdm(total = len(test_video_dir), desc="Test phase...")
    for idx, video in enumerate(test_video_dir):
       
        frame_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test/videos', video, '*.png')))
       
        ##################### 2D #############################
        frame_paths_2d = frame_paths[start_frame_index:]
       
        video_dataset_2d = USDataset2D(root=args.data_dir, img_paths=frame_paths_2d,
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS, split='test')

        video_loader_2d = DataLoader(dataset=video_dataset_2d, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)
        
        [test_f1_2d, test_acc_2d], [predictions_2d, targets_2d] = test(net_2d, video_loader_2d,
                                                                  checkpoint_dir=path_weights_2d,
                                                                  device=device, verbose=False)
        accuracy_2d.append(test_acc_2d)
       
        pred_2d_list.extend(predictions_2d)
        target_2d_list.extend(targets_2d)
        del video_dataset_2d, video_loader_2d

        
        ##################### 3D #############################
        video_clips = split_video_in_clips(frame_paths, args.clip_len)

        video_dataset_3d = USDataset3D(root=args.data_dir, clips_paths=video_clips,
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS)
        video_loader_3d = DataLoader(dataset=video_dataset_3d, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)




        
        [test_f1_3d, test_acc_3d], [predictions_3d, targets_3d] = test_3d(net_3d, video_loader_3d,
                                                                  checkpoint_dir=path_weights_3d,
                                                                  device=device, verbose=False)

        accuracy_3d.append(test_acc_3d)
        pred_3d_list.extend(predictions_3d)
        target_3d_list.extend(targets_3d)

        #del net_3d
        ##################### (2+1)D #############################

        
        [test_f1_2d1d, test_acc_2d1d], [predictions_2d1d, targets_2d1d] = test_3d(net_2d1d, video_loader_3d,
                                                                  checkpoint_dir=path_weights_2d1d,
                                                                  device=device, verbose=False)

        accuracy_2d1d.append(test_acc_2d1d)
        pred_2d1d_list.extend(predictions_2d1d)
        target_2d1d_list.extend(targets_2d1d)

        #del net_2d1d
        del video_dataset_3d, video_loader_3d
        pbar.update(1)
    pbar.close()

    print("Avg accuracy 2d: ", np.mean(accuracy_2d))
    print("Avg accuracy 3d: ", np.mean(accuracy_3d))
    print("Avg accuracy (2+1)d: ", np.mean(accuracy_2d1d))

    results = { 'accuracy 2D': round(np.mean(accuracy_2d) * 100, 3),
               'std 2D': round(np.std(accuracy_2d) * 100, 3),
               'accuracy 3D': round(np.mean(accuracy_3d) * 100, 3),
               'std 3D': round(np.std(accuracy_3d) * 100, 3),
               'accuracy (2+1)D': round(np.mean(accuracy_2d1d) * 100, 3),
               'std (2+1)D': round(np.std(accuracy_2d1d) * 100, 3)}
    

    target_names = ['class ' + str(c) for c in list(classes.values())]
    

    accuracy_per_class_2d = compute_accuracy_per_class(target_2d_list, pred_2d_list)
    accuracy_per_class_3d = compute_accuracy_per_class(target_3d_list, pred_3d_list)
    accuracy_per_class_2d1d = compute_accuracy_per_class(target_2d1d_list, pred_2d1d_list)


    accuracy_per_class = {'acc per class 2D': accuracy_per_class_2d,
                          'acc per class 3D': accuracy_per_class_3d,
                          'acc per class 2D1D': accuracy_per_class_2d1d}


    class_report_dict_2d = classification_report(torch.Tensor(target_2d_list), torch.Tensor(pred_2d_list), 
                                                 target_names=target_names, output_dict=True)
    class_report_dict_3d = classification_report(torch.Tensor(target_3d_list), torch.Tensor(pred_3d_list), 
                                                 target_names=target_names, output_dict=True)
    class_report_dict_2d1d = classification_report(torch.Tensor(target_2d1d_list), torch.Tensor(pred_2d1d_list), 
                                                 target_names=target_names, output_dict=True)
    class_report_results = {'2D report': class_report_dict_2d, '3D report': class_report_dict_3d, '(2+1)D': class_report_dict_2d1d}
    eval_list = [results, class_report_results, accuracy_per_class] 

    print("Accuracy per class: ", accuracy_per_class)
    

    fname_suffix = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, fname_suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    eval_file = os.path.join(log_dir, 'evaluation.json')
    with open(eval_file, "w+") as f:
        json.dump(eval_list, f, indent=2)


if __name__ == '__main__':
    main()