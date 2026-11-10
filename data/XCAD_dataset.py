import glob
import numpy as np
import os, os.path
import cv2
from torch.utils.data import Dataset
import data.util as Util
from PIL import Image

class XCADDataset(Dataset):
    def __init__(self, dataroot, split='train', source_dir=None):
        self.source_dir = source_dir
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot
        self.data_idx = -1

        if split == 'train':
            # 'trainB'= Image, 'trainC'=Background, 'trainA'=fractal label
            self.A_paths = sorted(glob.glob(os.path.join(dataroot, self.split, 'trainB', '*.png')))
            self.F_paths = sorted(glob.glob(os.path.join(dataroot, 'train', 'trainA', '*.png')))
            if source_dir:
                import json
                with open(os.path.join(dataroot, 'split_manifest.json')) as f:
                    manifest = json.load(f)
                self.A_paths = [os.path.join(source_dir, name) for name in manifest['train']]
            self.data_len = len(self.A_paths)
            self.data_lenF = len(self.F_paths)

        elif split == 'val':
            dataPath = os.path.join(dataroot, 'test', 'images')
            if not os.path.isdir(dataPath):
                # Fundus training has no labeled XCAD-style test split.
                # Keep the loader alive so training can start; Dice is skipped.
                self.imageNum = []
                self.data_len = 0
            else:
                dataFiles = sorted(os.listdir(dataPath))[:12]
                for isub, FileName in enumerate(dataFiles):
                    self.imageNum.append(FileName)
                self.data_len = len(self.imageNum)
        else:
            if 'XCAD' in dataroot:
                dataPath = os.path.join(dataroot, self.split, 'images')
                dataFiles = sorted(os.listdir(dataPath))
            elif 'DRIVE' in dataroot:
                dataPath = os.path.join(dataroot, 'train', 'images')
                dataFiles = sorted(os.listdir(dataPath))
            elif 'STARE' in dataroot:
                dataPath = os.path.join(dataroot, 'images')
                dataFiles = sorted(os.listdir(dataPath))

            for isub, FileName in enumerate(dataFiles):
                self.imageNum.append(FileName)
            self.data_len = len(self.imageNum)

        self.inputSize = 256

    def _random_subsample(self, data):
        opt1 = np.random.randint(0,2)
        opt2 = np.random.randint(0,2)
        data = data[opt1::2, opt2::2]
        return data
    
    def _random_crop(self, data):
        nh, nw = data.shape
        opt1, opt2 = 0, 0
        if nh > self.inputSize:
            opt1 = np.random.randint(0, nh-self.inputSize)
        if nw > self.inputSize:
            opt2 = np.random.randint(0, nw-self.inputSize)
        data = data[opt1:opt1+self.inputSize, opt2:opt2+self.inputSize]
        return data

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.data_len:
            self.data_idx = 0
            np.random.shuffle(self.A_paths)
            np.random.shuffle(self.F_paths)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.split == 'train':
            if not self.source_dir:
                self._shuffle_data_index()
            A_path = self.A_paths[index]
            F_path = self.F_paths[index % self.data_lenF]

            data_A = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE).astype('float')/255.
            data_F = cv2.imread(F_path, cv2.IMREAD_GRAYSCALE).astype('float')/255.

            if self.source_dir:
                # Crop excess black border; preserve aspect ratio before author subsampling.
                foreground = data_A > (10. / 255.)
                ys, xs = np.where(foreground)
                if len(xs):
                    data_A = data_A[ys.min():ys.max()+1, xs.min():xs.max()+1]
                h, w = data_A.shape
                side = max(h, w)
                data_A = cv2.copyMakeBorder(data_A, (side-h)//2, (side-h+1)//2, (side-w)//2, (side-w+1)//2, cv2.BORDER_CONSTANT, value=0)
                data_A = cv2.resize(data_A, (512, 512), interpolation=cv2.INTER_AREA)
            data_A = self._random_subsample(data_A)
            data_F = self._random_subsample(data_F)
        else:
            if self.data_len == 0:
                raise IndexError('No validation images for this dataset')
            dataInfo = self.imageNum[index]
            dataPath = glob.glob(os.path.join(self.dataroot, 'test', 'images', dataInfo[:-4]+'*'))[0]
            data_A = cv2.imread(dataPath, cv2.IMREAD_GRAYSCALE).astype('float') / 255.
            labelPath = glob.glob(os.path.join(self.dataroot, 'test', 'masks', dataInfo[:-4]+'*'))[0]
            data_F = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE).astype('float') / 255.

            A_path = dataInfo
        [data_A, data_F] = Util.transform_augment([data_A, data_F], split=self.split, min_max=(-1, 1))

        return {'A': data_A, 'F': data_F, 'P':A_path, 'Index': index}
