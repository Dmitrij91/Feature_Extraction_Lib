import torch
import numpy as np
from PIL import Image
from torch.utils import data
from skimage.util import view_as_windows
import os


def get_first_training_files(path_dir,train=True):
    if train:
        path_to_train_data = os.path.join(path_dir,"training_data.csv")
        dset_files = read_Image_list(path_to_train_data)
    else:
        path_to_test_data = os.path.join(path_dir,"testing_data.csv")
        dset_files = read_Image_list(path_to_test_data)
    assert len(dset_files) > 0, "No valid volume data was provided."
    rel_path = dset_files[1]
    path_train_im = rel_path[1][:-4]
    path_label_im = rel_path[2][:-4]
    return path_train_im, path_label_im

def build_data_loader(size_window, c,path_dir, batch_size=32, debug=0, workers=4, single_pixel_out=True, train=True):
    """
    size_window:      Spatial shape of input data to model.
    debug:            In debug mode, only part of the data is loaded.
    single_voxel_out: Output is a single pixel label. This is in contrast to fully convolutional models which require a patch of pixels.
    """
    if not debug == 1:
        dataset = Multiple_Image_Patch_Dataset(size_window, c,path_dir ,single_pixel_out=single_pixel_out, train=train)
    else:
        fname, gt_fname = get_first_training_files(path_dir,train)
        dataset = Image_Patch_Dataset(fname, gt_fname, size_window, c, single_pixel_out=single_pixel_out)
    return data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

class Multiple_Image_Patch_Dataset(data.Dataset):
    """
    Wraps multiple Image data files instances.
    """
    def __init__(self, size_window, c,path_dir, single_pixel_out=True, train=True):
        """
        """
        self.dsets = []

        if train:
            path_to_train_data = os.path.join(path_dir,"training_data.csv")
            dset_files = read_Image_list(path_to_train_data)
        else:
            path_to_test_data = os.path.join(path_dir,"testing_data.csv")
            dset_files = read_Image_list(path_to_test_data)

        for path_train_im,path_label_im in dset_files[1:]:
            assert path_train_im.endswith(".png")
            self.dsets.append(Image_Patch_Dataset(path_train_im, path_label_im, size_window, c, single_pixel_out=single_pixel_out))

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

class Image_Patch_Dataset(data.Dataset):
    """
    Loads training data from a single Image.
    """
    def __init__(self, data_path, labels_path, size_window, c, single_pixel_out=True):
        """
        """
        mask = np.array([0.299,0.587,0.114])/(255.0)
        self.raw_data = np.einsum('ijk,k->ij',np.array(Image.open(data_path).copy()).astype(np.double)[:,:,:],mask)
        self.size_window = size_window
        try:
            self.view = view_as_windows(self.raw_data, size_window)
        except:
            print(f"Failed to stride image {data_path} with shape {self.raw_data.shape} using window of size {size_window}. Skipping.")
            return
        self.view_spatial_shape = self.view.shape[:2]
        self.labels = np.array(Image.open(labels_path).copy())
        
        c_dat = int(self.labels.max()) + 1
        
        assert self.labels.min() == 0
        assert self.labels.max() == c-1
        assert not np.isnan(np.sum(self.labels))
        assert not np.isnan(np.sum(self.raw_data))
        
        self.single_pixel_out = single_pixel_out
        if single_pixel_out:
            rx, ry = (size_window[0] - 1)//2, (size_window[1] - 1)//2
            self.labels = self.labels[rx:-rx,ry:-ry]
            assert self.labels.shape == self.view_spatial_shape
        else:
            self.label_view = view_as_windows(self.labels, size_window)

    def __len__(self):
        return np.prod(self.view_spatial_shape)
    
    def __getitem__(self, idx):
        (x,y) = np.unravel_index(idx, self.view_spatial_shape)

        patch = self.view[x,y,:,:].reshape(self.size_window)
        if self.single_pixel_out:
            patch = patch.reshape((self.size_window))
            return torch.FloatTensor(patch), torch.LongTensor(self.labels[x,y].reshape((1,)))
        labels = self.label_view[x,y,:,:]
        return torch.FloatTensor(patch), torch.LongTensor(labels.reshape(self.size_window))

def read_Image_list(path):
    with open(path, "r") as f:
        lines = f.readlines()
    fpaths = [fn[:-1].split(',')[1:] if fn.endswith("\n") else fn for fn in lines]
    return fpaths