import os
import glob
import json
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse


def non_redundant_indices(imgs: np.ndarray, mask: np.ndarray):
    """
    Return the frame indices of a video sequence (imgs) where the SSIM is lower than the average SSIM threshold
    in the sequence. Such indices represent non-redundant frames, i.e. not too similar to others in the sequence.
    Note: the SSIM is computed only on the box surrounding the real data in the US image, trying to avoid
    the black surrounding as much as possible.
    """

    first_x = np.where(np.argmax(mask, axis=0) > 0)[0][0]
    first_y = np.where(np.argmax(mask, axis=1) > 0)[0][0]
    last_x = 1200 - np.where(np.argmax(mask[:, ::-1], axis=0) > 0)[0][0]
    last_y = 760 - np.where(np.argmax(mask[::-1], axis=1) > 0)[0][0]

    ssims = np.zeros(len(imgs))
    prev_frame = imgs[0]
    for k, frame in enumerate(imgs[1:]):
        ssim_frame = ssim(prev_frame[first_y:last_y, first_x:last_x],
                          frame[first_y:last_y, first_x:last_x],
                          data_range=255)
        prev_frame = frame.copy()
        ssims[k] = ssim_frame

    ssim_thr = np.mean(ssims)
    indices = np.where(ssims < ssim_thr)[0]
    indices = np.array(list(set([0] + list(indices) + list(indices+1))), dtype=int)
    return indices[indices < len(imgs)]


def ecomask(imgs: np.ndarray, margin: int = 20):
    """
    Find the US mask from a sequence of US frames. The mask is True where actual data is present in the US image
    and False in the black surrounding.
    """
    mask = imgs.mean(axis=0).astype(bool)
    mask[:, :40] = 0
    maskmask = np.pad(mask, margin)

    maskmask = (np.abs(np.diff(maskmask, append=1, axis=0)) + np.abs(np.diff(maskmask, append=1, axis=1))).astype(bool)
    y, x = np.where(maskmask)
    notedges = (y > 0) & (y < maskmask.shape[0] - 1) & (x > 0) & (x < maskmask.shape[1] - 1)
    y, x = y[notedges], x[notedges]
    for k in range(1, margin):
        maskmask[y - k, x - k] = 1
        maskmask[y + k, x + k] = 1
        maskmask[y + k, x - k] = 1
        maskmask[y - k, x + k] = 1
        maskmask[y - k, x] = 1
        maskmask[y, x - k] = 1
        maskmask[y + k, x] = 1
        maskmask[y, x + k] = 1
    maskmask = maskmask[margin:-margin, margin:-margin]
    mask[maskmask] = 0
    return mask


class USDatasetAdaptiveSubsample(Dataset):
    """
    Load each video from a given train/test set and return the sequence of non-redundant frames within it,
    together with all their labels and full paths of both images and labels.
    """

    def __init__(self, root: str, downsample: bool = True):
        assert os.path.exists(os.path.join(root, 'videos')) and os.path.exists(os.path.join(root, 'labels'))
        video_paths = sorted(glob.glob(os.path.join(root, 'videos', '*'))) 
        label_paths = sorted(glob.glob(os.path.join(root, 'labels', '*'))) 
        assert len(video_paths) == len(label_paths)

        self.vid_paths = list(map(lambda p: sorted(glob.glob(os.path.join(p, '*.png'))), video_paths))  
        self.lbl_paths = list(map(lambda p: sorted(glob.glob(os.path.join(p, '*.txt'))), label_paths))  

        with open(os.path.join(os.path.split(root)[0], 'classes.json'), 'r') as f:
            self.classes = json.load(f)
        self.num_classes = len(self.classes)

        self.downsample = downsample

        self.basic_transforms = transforms.Compose([transforms.PILToTensor(),                           
                                                    transforms.Grayscale(num_output_channels=1)]) 

    def __getitem__(self, index):
        vid_path = np.array(self.vid_paths[index], dtype=str)
        lbl_path = np.array(self.lbl_paths[index], dtype=str)

        imgs, lbls = [], []
        for i in range(len(vid_path)):
            with open(lbl_path[i], 'r') as f:
                lbl = self.classes[f.read()]
            img = self.basic_transforms(default_loader(vid_path[i])).numpy()[0]   
            imgs.append(img)
            lbls.append(lbl)
        imgs = np.stack(imgs).astype(np.uint8)
        lbls = np.stack(lbls).astype(int)

        indices = range(len(lbls))
        if self.downsample:
            mask4eco = ecomask(imgs, margin=10)
            indices = non_redundant_indices(imgs, mask4eco)

        return (imgs[indices], vid_path[indices]), (lbls[indices], lbl_path[indices])

    def __len__(self):
        return len(self.vid_paths)


def populate_2d_split(split_dir: str, res_dir: str, split: str = 'test'):
    
    data_dir = os.path.join(split_dir, split)
    dataset_vid = USDatasetAdaptiveSubsample(data_dir, downsample=True)   

    
    classes = dataset_vid.classes
    lbl_paths = []
    for lbl in classes.values():
        lpath = os.path.join(res_dir, split, str(lbl))
        lbl_paths.append(lpath)
        os.makedirs(lpath, exist_ok=True)

    num_frames = []
    for index in tqdm(range(len(dataset_vid)), desc='Progress:'):
        # Retrieve the time-subsampled image sequence x_vid and image labels y_vid, together with their paths
        (x_vid, x_paths), (y_vid, y_paths) = dataset_vid[index]

        
        num_frames.append(len(y_vid))

        
        for k in range(len(x_paths)):
            lbl = y_vid[k]
            old_path = x_paths[k]
            new_path = os.path.join(res_dir, split, str(lbl), os.path.split(old_path)[-1])
            shutil.copy(old_path, new_path)

    
    with open(os.path.join(res_dir, split + '-nframes.pickle'), 'wb') as f:
        pickle.dump(num_frames, f)

    
    frames_in_label = np.zeros(len(classes), dtype=int)
    for k, lbl_path in enumerate(lbl_paths):
        n = len(glob.glob(os.path.join(lbl_path, '*.png')))
        frames_in_label[k] = n
    np.save(os.path.join(split_dir, split + '-class-number.npy'), frames_in_label)

    
    plt.figure()
    plt.bar(x=range(len(classes)), height=frames_in_label / frames_in_label.sum() * 100)
    plt.ylim(0, 60)
    plt.savefig(os.path.join(split_dir, split + '-class-distribution.svg'))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-split_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-res_dir', type=str, help='Directory where the 2D images will be stored.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    split_dir = args.data_dir

   
    res_dir = args.res_dir
    if os.path.exists(res_dir):
        try:
            shutil.rmtree(res_dir)
            print(f"Directory '{res_dir}' and its contents successfully deleted.")
        except OSError as e:
            print(f"Error: {e}")
    os.makedirs(res_dir, exist_ok=True)

    
    prev_dir = os.path.split(split_dir)[0]
    shutil.copyfile(os.path.join(prev_dir, 'exams.json'), os.path.join(res_dir, 'exams.json'))
    shutil.copyfile(os.path.join(prev_dir, 'summary', 'tracks.csv'), os.path.join(res_dir, 'tracks.csv'))
    shutil.copyfile(os.path.join(prev_dir, 'summary', 'infos.csv'), os.path.join(res_dir, 'infos.csv'))
    shutil.copyfile(os.path.join(split_dir, 'train-infos.csv'), os.path.join(res_dir, 'train-infos.csv'))
    shutil.copyfile(os.path.join(split_dir, 'test-infos.csv'), os.path.join(res_dir, 'test-infos.csv'))
    shutil.copyfile(os.path.join(split_dir, 'classes.json'), os.path.join(res_dir, 'classes.json'))

    
    with open(os.path.join(split_dir, 'classes.json'), 'r') as f:
        classes = json.load(f)

    print('\n' + '----'*5 + ' Training set ' + '----'*5)
    populate_2d_split(split_dir, res_dir, 'train')
    print('Done!')

    print('\n' + '----'*5 + ' Test set ' + '----'*5)
    populate_2d_split(split_dir, res_dir, 'test')
    print('Done!')
