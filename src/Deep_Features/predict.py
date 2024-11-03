import numpy as np
import torch
import os.path
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Use deep network classifier for oct feature extraction.')
parser.add_argument("data_filepath",
    type=str,
    help="Path of input data file.")
parser.add_argument("model_filepath",
    type=str,
    help="Path of model checkpoint.")
parser.add_argument("--batch_size",
    type=int,
    help="Size of batches used in inference.",
    default=1024)
parser.add_argument("--legacy_kernel",
    type=int,
    help="Use legacy kernel size (large kernel in 3D resnet encoder).",
    default=0)
parser.add_argument("--dist_outname",
    type=str,
    help="Path of distance matrix file to be created.",
    default="distance.npy")
parser.add_argument("--overlap_windows",
    type=int,
    help="Number of pixels patches overlap in fully convolutional predictions.",
    default=4)
args = parser.parse_args()

assert os.path.isfile(args.model_filepath)
#assert os.path.isfile(args.data_filepath)
assert args.model_filepath.endswith(".pt")
assert args.data_filepath.endswith(".npy")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load model config
with open(os.path.join(os.path.dirname(args.model_filepath), "training_config.json"), "r") as config_file:
    config = json.loads(config_file.read())
if not 'model' in config.keys():
    config['model'] = 'resnet'

if "num_layers" in config:
    c = config["num_layers"]
else:
    c = 14

fully_convolutional = (config['model'] in ['unet'])
patch_3d = ("x" in config['window_size'])

if config['model'] == "resnet":
    if patch_3d:
        from model_resnet import resnet
        if args.legacy_kernel == 1:
            model = resnet(config['model_size'], 1, c, dropout=config['dropout'], kernel_size_enc=7)
        else:
            model = resnet(config['model_size'], 1, c, dropout=config['dropout'])
    else:
        from model_resnet_2d import resnet
        model = resnet(config['model_size'], 1, c, dropout=config['dropout'])
elif config['model'] == "dense":
    from model_dense import dense_predictor
    input_dim = config['window_size']['y']*config['window_size']['z']
    if patch_3d:
        input_dim *= config['window_size']['x']
    model = dense_predictor(input_dim, c, config['model_size'], activation=config['activation'])
elif config['model'] == "flat_dense":
    from model_dense import flat_dense_predictor
    input_dim = config['window_size']['y']*config['window_size']['z']
    if patch_3d:
        input_dim *= config['window_size']['x']
    model = flat_dense_predictor(input_dim, c, config['model_size'], activation=config['activation'])
elif config['model'] == 'unet':
    from model_unet import unet_predictor
    model = unet_predictor(size=config['model_size'])
else:
    print(f"Unknown model for checkpoint {args.model_filepath}.")
    exit()
model.load_state_dict(torch.load(args.model_filepath))
model.to(device)

vol = np.load(args.data_filepath)
from skimage.util import view_as_windows, view_as_blocks
if patch_3d:
    window_size = (config['window_size']['x'],config['window_size']['y'],config['window_size']['z'])
else:
    window_size = (config['window_size']['y'],config['window_size']['z'])

model.eval()
if fully_convolutional:
    assert 2*args.overlap_windows < min(*window_size)
    inner_window_size = tuple([ws-2*args.overlap_windows for ws in window_size])
    # crop vol to make it compatible with windowig stride
    block_count = [(vs - 2*args.overlap_windows) // iws for vs, iws in zip(vol.shape, inner_window_size)]
    vol_shape = tuple([b*iws + 2*args.overlap_windows for b, iws in zip(block_count, inner_window_size)])
    vol = vol[:vol_shape[0],:vol_shape[1],:vol_shape[2]]
    
    # gather windows for prediction
    patches = view_as_windows(vol, window_size, step=inner_window_size).reshape((-1,1,*window_size))
    cropped_vol_shape = tuple([vs - 2*args.overlap_windows for vs in vol.shape])
    scores = np.empty((*cropped_vol_shape, c), dtype=np.float32)
    scores_view = view_as_blocks(scores, (*inner_window_size, c))
    
    for batch_ind in tqdm(range(1 + patches.shape[0] // args.batch_size)):
        batch_input = torch.FloatTensor(patches[batch_ind*args.batch_size:(batch_ind+1)*args.batch_size,...]).to(device)
        if batch_input.shape[0] == 0:
            continue
        predictions = model(batch_input).data.cpu().numpy()

        for i, ind in enumerate(range(batch_ind*args.batch_size, (batch_ind+1)*args.batch_size)):
            if i < predictions.shape[0]:
                block_ind = np.unravel_index(ind, scores_view.shape[:3])
                scores_view[block_ind[0],block_ind[1],block_ind[2],0,...] = np.moveaxis(predictions[i,:,args.overlap_windows:-args.overlap_windows,args.overlap_windows:-args.overlap_windows,args.overlap_windows:-args.overlap_windows], 0, -1)
else:
    if patch_3d:
        patches = view_as_windows(vol, window_size)
    else:
        patches = view_as_windows(vol, (1, *window_size))
    patches_shape = patches.shape
    scores = np.empty((patches_shape[0], int(np.prod(patches_shape[1:3])), c), dtype=np.float32)

    for bscan in tqdm(range(patches_shape[0])):
        b_patches = patches[bscan,...].reshape((-1,1,*window_size))

        for batch_ind in range(1 + b_patches.shape[0] // args.batch_size):
            batch_input = torch.FloatTensor(b_patches[batch_ind*args.batch_size:(batch_ind+1)*args.batch_size,...]).to(device)
            if batch_input.shape[0] == 0:
                continue
            predictions = model(batch_input)
            current_batch_size = batch_input.shape[0]
            scores[bscan,batch_ind*args.batch_size:(batch_ind+1)*args.batch_size,:] = predictions.data.cpu().numpy().reshape((current_batch_size, c))
    scores = scores.reshape((*patches_shape[:3],c))

# normalize distances
dist = - scores
dist -= dist.min()
dist /= dist.max()
dist *= 10.0

np.save(args.dist_outname, dist)
print(f"Distance matrix with shape {scores.shape} created at {args.dist_outname}")
