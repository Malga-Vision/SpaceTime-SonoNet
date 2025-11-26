from sononet2d.models import SonoNet2D
from sononet3d.models import SonoNet3D, SonoNet3D_2_1d
import os
import random
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


IMG_SIZE = [256, 256]
NUM_CHANNELS = 1 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Save plot directory.')
    parser.add_argument('-model_dir_2d', type=str, help='Directory of the 2d model to load.')
    parser.add_argument('-model_dir_3d', type=str, help='Directory of the 3d model to load.')
    parser.add_argument('-model_dir_2_1d', type=str, help='Directory of the (2+1)d model to load.')
    parser.add_argument('-num_features', type=int, default=32,
                        help='Number of hidden features for the SonoNet model: either 16, 32 or 64. Default is 16.')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpu', nargs='*', type=int, default=[0], help='Choose the device number of your GPU. Default is 0.')
    parser.add_argument('-clip_len', type=int, default=10, help='Number of frame in a clip. Default is 10.')
    parser.add_argument('-video_name', type=str, default=None, help='Video name to test. If None, a random video will be chosen from the test set.')
    
    return parser.parse_args()

def get_video_and_label(data_dir, video_name: None):
    video_folder = os.path.join(data_dir, 'videos')
    labels_folder = os.path.join(data_dir, 'labels')

    if video_name:
        video_path = os.path.join(video_folder, video_name)
        label_path = os.path.join(labels_folder, video_name)

    else:
        video_paths = os.listdir(video_folder)
        random_video = random.choice(video_paths)
        video_path = os.path.join(video_folder, random_video)
        label_path = os.path.join(labels_folder, random_video)

    return video_path, label_path

def split_video_in_clips(frame_path, clip_len):
    clips = [frame_path[i:i+clip_len]  for i in range(len(frame_path)) if len(frame_path[i:i+clip_len]) == clip_len]

    return clips

def count_samples_per_class(clips, classes):
    target_classes = list(classes.values())
    class_counts_dict = {class_num: 0 for class_num in target_classes}
    for clip in clips:
        with open(clip[-1].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:
            lbl = int(classes[f.read()])
        class_counts_dict[lbl] += 1
    return list(class_counts_dict.values())


def plot_comparison(start_frame_index, target_labels, clip_pred_labels, clip_pred_2_1d_label, frame_pred_labels, title, log_dir, acc_2d, acc_3d, acc_2_1d):
    indices = list(range(len(target_labels)))
    indices = [i + start_frame_index for i in indices]

    clip_pred_labels = [x + 0.12 for x in clip_pred_labels]
    clip_pred_2_1d_label = [x for x in clip_pred_2_1d_label]
    frame_pred_labels = [x - 0.12 for x in frame_pred_labels]

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)

    grouped_rects = []
    start = 0
    for i in range(1, len(target_labels)):
        if target_labels[i] != target_labels[i - 1]:
            grouped_rects.append((indices[start], indices[i - 1], target_labels[i - 1]))
            start = i
    
    grouped_rects.append((indices[start], indices[-1], target_labels[-1]))

    for start_idx, end_idx, label in grouped_rects:
        width = end_idx - start_idx + 1
        rect = Rectangle((start_idx, label - 0.3), width, 0.6,
                         color='green', alpha=0.2, zorder=2)
        ax.add_patch(rect)

    
    ax.scatter(indices, clip_pred_labels, marker='x', s=10, color='red', label=f'3D prediction ({round(acc_3d*100, 2)}%)', zorder=3)
    ax.scatter(indices, clip_pred_2_1d_label, marker='x', s=10, color="#466108", label=f'(2+1)D prediction ({round(acc_2_1d*100,2)}%)', zorder=3)
    ax.scatter(indices, frame_pred_labels, marker='x', s=10, color="blue", label=f'2D model prediction ({round(acc_2d*100,2)}%)', zorder=3)

   
    target_patch = Rectangle((0, 0), 1, 1, color='green', alpha=0.2, label='Target Labels')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [target_patch], fontsize=12)

    if not os.path.exists(os.path.join(log_dir, 'accuracy')):
        os.makedirs(os.path.join(log_dir, 'accuracy'))

    plt.xlabel('Frame Indices', fontsize=20)
    plt.ylabel('Scanplane Labels', fontsize=20)
    plt.savefig(os.path.join(log_dir, 'accuracy', title + '.png'), dpi=300, bbox_inches='tight')
    plt.show()



class Normalize:
    """Perform normalization and fill NaNs with zeros."""

    def __init__(self):
        pass

    def __call__(self, image):
        lastdim = len(image.size()) - 1
        image /= torch.amax(image, dim=(lastdim-1, lastdim))[..., None, None]
        return torch.where(torch.isnan(image), torch.zeros_like(image), image)

class USDataset2D(Dataset):

    def __init__(self, root: str, img_paths, in_channels: int = 1, 
                 img_size: List[int] = [256, 256]):
        
        self.img_paths = img_paths
        with open(os.path.join(root, 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        
        
        self.num_channels = in_channels
        self.img_size = img_size

        self.basic_transforms = transforms.Compose(
            [transforms.PILToTensor(), transforms.Resize(img_size, antialias=True), transforms.Grayscale(num_output_channels=self.num_channels), transforms.ConvertImageDtype(dtype=torch.float), Normalize()] 
        )
        
        self.num_samples = len(self.img_paths)
        self.num_classes = len(self.classes)
        
    def __getitem__(self, index):
        img = self.basic_transforms(default_loader(self.img_paths[index]))
        with open(self.img_paths[index].replace('videos', 'labels')[:-4] + '.txt', 'r') as f:
            lbl = int(self.classes[f.read()])
        return img, lbl
                              
    def __len__(self):
        return self.num_samples
    

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
            [transforms.PILToTensor(), transforms.Grayscale(num_output_channels=self.num_channels), 
             transforms.ConvertImageDtype(dtype=torch.float), 
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
    random_video_path, random_label_path = get_video_and_label(os.path.join(args.data_dir, 'test'), video_name=args.video_name)
    frame_paths = sorted(glob.glob(os.path.join(random_video_path, '*.png')))

    frame_paths_2d = frame_paths[start_frame_index:]
    
    video_dataset_2d = USDataset2D(root=args.data_dir, img_paths=frame_paths_2d,
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS)

    video_loader_2d = DataLoader(dataset=video_dataset_2d, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    net_2d = SonoNet2D(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=video_dataset_2d.num_classes, train_classifier_only=False)
    net_2d.to(device)
    path_weights_2d =  args.model_dir_2d

    print('\nStarting test phase...')
    [test_f1_3d, test_acc_2d], [predictions_2d, targets_2d] = test(net_2d, video_loader_2d,
                                                                  checkpoint_dir=path_weights_2d,
                                                                  device=device)
    print("Test accuracy: ", test_acc_2d)
    del net_2d


    ######################################### 3D ####################################
    video_clips = split_video_in_clips(frame_paths, args.clip_len)

    video_dataset_3d = USDataset3D(root=args.data_dir, clips_paths=video_clips,
                                      img_size=IMG_SIZE, in_channels=NUM_CHANNELS)
    video_loader_3d = DataLoader(dataset=video_dataset_3d, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    
    net_3d = SonoNet3D(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=video_dataset_3d.num_classes, train_classifier_only=False)
    net_3d.to(device)
    path_weights_3d =  args.model_dir_3d


    
    print('\nStarting test phase...')
    [test_f1_3d, test_acc_3d], [predictions_3d, targets_3d] = test_3d(net_3d, video_loader_3d,
                                                                  checkpoint_dir=path_weights_3d,
                                                                 device=device)
    del net_3d
    print("Test accuracy: ", test_acc_3d)
    ########################################## (2+1)D ####################################
    net_2_1d = SonoNet3D_2_1d(in_channels=NUM_CHANNELS, hid_features=args.num_features,
                out_labels=video_dataset_3d.num_classes, train_classifier_only=False)
    net_2_1d.to(device)
    path_weights_2_1d =  args.model_dir_2_1d
    print('\nStarting test phase...')
    [test_f1_2_1d, test_acc_2_1d], [predictions_2_1d, targets_2_1d] = test_3d(net_2_1d, video_loader_3d,
                                                                  checkpoint_dir=path_weights_2_1d,
                                                                  device=device)
    print("Test accuracy: ", test_acc_2_1d)

    title = os.path.split(random_video_path)[1]
    print("Video: ", title)
    plot_comparison(start_frame_index, targets_3d, predictions_3d, predictions_2_1d, predictions_2d, title, args.log_dir, test_acc_2d, test_acc_3d, test_acc_2_1d)


    



if __name__ == '__main__':
    main()
