from util.layers import join_layers
from util.path import read_vol_list
import torch
import numpy as np
import os
from torch.utils import data
from skimage.util import view_as_windows
import glob

def get_first_training_files(data_path, train=True):
    if train:
        dset_files = read_vol_list("training_vols.csv")
    else:
        dset_files = read_vol_list("testing_vols.csv")
    assert len(dset_files) > 0, "No valid volume data was provided."
    base_path = os.path.join(data_path, dset_files[0][:-4])
    fname = f"{base_path}.npy"
    gt_fname = f"{base_path}_gt.npy"
    return fname, gt_fname

def build_data_loader(data_path, size_window, c, batch_size=64, debug=0, workers=4, single_voxel_out=True, train=True):
    """
    data_path:        Load data from here.
    size_window:      Spatial shape of input data to model.
    debug:            In debug mode, only part of the data is loaded.
    single_voxel_out: Output is a single voxel label. This is in contrast to fully convolutional models which require a patch of voxels.
    """
    if not debug == 1:
        dataset = OCTPatchMultifileDataset(data_path, size_window, c, single_voxel_out=single_voxel_out, train=train)
    else:
        fname, gt_fname = get_first_training_files(data_path, train)
        dataset = OCTPatchDataset(fname, gt_fname, size_window, c, single_voxel_out=single_voxel_out)
    return data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

class OCTPatchMultifileDataset(data.Dataset):
    """
    Wraps multiple OCTPatchDataset instances, each pertaining to a single volume.
    """
    def __init__(self, data_dirpath, size_window, c, single_voxel_out=True, train=True):
        """
        """
        self.dsets = []

        if train:
            dset_files = read_vol_list("training_vols.csv")
        else:
            dset_files = read_vol_list("testing_vols.csv")

        for rel_path in dset_files:
            assert rel_path.endswith(".mat") or rel_path.endswith(".vol")
            base_path = os.path.join(data_dirpath, rel_path[:-4])
            fname = f"{base_path}.npy"
            gt_fname = f"{base_path}_gt.npy"
            self.dsets.append(OCTPatchDataset(fname, gt_fname, size_window, c, single_voxel_out=single_voxel_out))

    def __len__(self):
        return np.sum([len(ds) for ds in self.dsets])

    def __getitem__(self, idx):
        """
        """
        for ds in self.dsets:
            if idx >= len(ds):
                idx -= len(ds)
            else:
                return ds[idx]

class OCTPatchDataset(data.Dataset):
    """
    Loads training data from a single OCT volume.
    """
    def __init__(self, data_path, labels_path, size_window, c, single_voxel_out=True):
        """
        """
        self.raw_data = np.load(data_path)
        self.size_window = size_window
        try:
            self.view = view_as_windows(self.raw_data, (1, *size_window))
        except:
            print(f"Failed to stride volume {data_path} with shape {self.raw_data.shape} using window of size {size_window}. Skipping.")
            self.view_spatial_shape = (0,0,0)
            return
        self.view_spatial_shape = self.view.shape[:3]
        self.labels = np.load(labels_path)
        
        c_dat = int(self.labels.max()) + 1
        assert c_dat in [12, 14]
        if c == 12 and c_dat == 14:
            self.labels = join_layers(self.labels)
        
        assert self.labels.min() == 0
        assert self.labels.max() == c-1
        assert not np.isnan(np.sum(self.labels))
        assert not np.isnan(np.sum(self.raw_data))
        
        self.single_voxel_out = single_voxel_out
        if single_voxel_out:
            ry, rz = (size_window[0] - 1)//2, (size_window[1] - 1)//2
            self.labels = self.labels[:,ry:-ry,rz:-rz]
            assert self.labels.shape == self.view_spatial_shape
        else:
            self.label_view = view_as_windows(self.labels, (1, *size_window))

    def __len__(self):
        return np.prod(self.view_spatial_shape)
    
    def __getitem__(self, idx):
        (x,y,z) = np.unravel_index(idx, self.view_spatial_shape)

        patch = self.view[x,y,z,:,:,:].reshape(self.size_window)
        if self.single_voxel_out:
            patch = patch.reshape((1,*self.size_window))
            return torch.FloatTensor(patch), torch.LongTensor(self.labels[x,y,z].reshape((1,)))
        labels = self.label_view[x,y,z,:,:,:]
        return torch.FloatTensor(patch), torch.LongTensor(labels.reshape(self.size_window))