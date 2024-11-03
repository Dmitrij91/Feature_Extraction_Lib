import sys, os
base_path = os.path.abspath("..")
if not base_path in sys.path:
    sys.path.append(base_path)

import numpy as np
import argparse
import json
from shutil import copyfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Train deep network classifier for oct feature extraction.')
parser.add_argument("--resume_hash",
    type=str,
    help="Hash of previous training instance.",
    default="")
parser.add_argument("--debug",
    type=int,
    help="Run training in debug mode. No logs are created and limited data is loaded.",
    default=0)
args = parser.parse_args()

config_path = '../training_config.json'
if not os.path.isfile(config_path):
    print("Please provide training config.")
    exit()
with open(config_path, "r") as config_file:
    config = json.loads(config_file.read())

# check how many layers to segment
if "num_layers" in config:
    c = config["num_layers"]
else:
    c = 14

# hash to identify this training instance with
if not len(args.resume_hash) == 0:
    training_hash = args.resume_hash
else:
    import hashlib
    import time
    training_hash = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())

patch_3d = ('x' in config['window_size'])

if patch_3d:
    window_str = f"{config['window_size']['x']}x{config['window_size']['y']}x{config['window_size']['z']}"
else:
    window_str = f"{config['window_size']['y']}x{config['window_size']['z']}"
model_identifier = f"{config['model']}{config['model_size']}_{training_hash}"

model_savedir = f"../results/trained_models/{model_identifier}"
if (not os.path.isdir(model_savedir)) and (not args.debug == 1):
    os.mkdir(model_savedir)
model_savepath = os.path.join(model_savedir, f"{model_identifier}.pt")

summary_logdir = f"./runs/{model_identifier}"
if (not os.path.isdir(summary_logdir)) and (not args.debug == 1):
    os.mkdir(summary_logdir)

distance_matrix_dir = f"../results/distance_matrices/{model_identifier}"
if (not os.path.isdir(distance_matrix_dir)) and (not args.debug == 1):
    os.mkdir(distance_matrix_dir)

segmentations_dir = f"../results/segmentations/{model_identifier}"
if (not os.path.isdir(segmentations_dir)) and (not args.debug == 1):
    os.mkdir(segmentations_dir)

# copy config file to places one might refer to
if not args.debug == 1:
    copyfile(config_path, os.path.join(model_savedir, "training_config.json"))
    copyfile(config_path, os.path.join(summary_logdir, "training_config.json"))
    copyfile(config_path, os.path.join(distance_matrix_dir, "training_config.json"))
    copyfile(config_path, os.path.join(segmentations_dir, "training_config.json"))


if patch_3d:
    size_window = (config['window_size']['x'], config['window_size']['y'], config['window_size']['z'])
else:
    size_window = (config['window_size']['y'], config['window_size']['z'])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# build model
if config['model'] == "resnet":
    if patch_3d:
        from model_resnet import resnet
        model = resnet(config['model_size'], 1, c, dropout=config['dropout'])
    else:
        from model_resnet_2d import resnet
        model = resnet(config['model_size'], 1, c, dropout=config['dropout'])
elif config['model'] == "linear":
    from model_linear import linear_predictor
    model = linear_predictor(int(np.prod(size_window)), c)
elif config['model'] == "dense":
    from model_dense import dense_predictor
    model = dense_predictor(int(np.prod(size_window)), c, size=config['model_size'], activation=config['activation'], dropout=config['dropout'])
elif config['model'] == "flat_dense":
    from model_dense import flat_dense_predictor
    model = flat_dense_predictor(int(np.prod(size_window)), c, size=config['model_size'], activation=config['activation'], dropout=config['dropout'])
elif config['model'] == "unet":
    assert patch_3d, "Only 3D patches supported in U-net architecture"
    from model_unet import unet_predictor
    model = unet_predictor(size=config['model_size'])
else:
    print(f"Unknown model {config['model']}.")
    exit()

# resume training if possible
if os.path.isfile(model_savepath):
    model.load_state_dict(torch.load(model_savepath))

fully_convolutional = (config['model'] in ['unet'])
model.to(device)

if patch_3d:
    from dataloader import build_data_loader
else:
    from dataloader_2d import build_data_loader

train_dataloader = build_data_loader(
    '../data/oct_converted', size_window, c,
    train=True,
    batch_size=config['batch_size'],
    debug=args.debug,
    workers=config['workers_training'],
    single_voxel_out=(not fully_convolutional))
test_dataloader = build_data_loader('../data/oct_converted',
    size_window, c,
    train=False,
    batch_size=config['batch_size'],
    debug=args.debug,
    workers=config['workers_testing'],
    single_voxel_out=(not fully_convolutional))
test_iter = iter(test_dataloader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if not args.debug == 1:
    writer = SummaryWriter(summary_logdir)

log_interval = int(1e3)
save_interval = int(1e4)
num_test_batches = 50

running_loss = 0.0
for epoch in range(100):  # loop over the dataset multiple times

    for i, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if fully_convolutional:
            # add channel dimension
            inputs = inputs.unsqueeze(1)
        else:
            labels = labels.flatten()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        if fully_convolutional:
            # remove channel dimension
            outputs = torch.squeeze(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_interval == log_interval-1:
            print(f"batch {i} training loss {running_loss / log_interval}")
            if not args.debug == 1:
                writer.add_scalar('training_loss', running_loss / log_interval, epoch*len(train_dataloader) + i)
            running_loss = 0.0

            # run testing
            model.eval()
            test_loss = 0.0
            for test_batches in range(num_test_batches):
                try:
                    test_data, test_labels = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_dataloader)
                    test_data, test_labels = next(test_iter)
                if fully_convolutional:
                    test_data = test_data.unsqueeze(1)
                else:
                    test_labels = test_labels.flatten()
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)
                test_out = model(test_data)
                if fully_convolutional:
                    test_out = torch.squeeze(test_out, 1)
                test_loss += criterion(test_out, test_labels).item()
            print(f"batch {i} testing loss {test_loss / num_test_batches}")
            if not args.debug == 1:
                writer.add_scalar('testing_loss', test_loss / num_test_batches, epoch*len(train_dataloader) + i)
            model.train()

        if i % save_interval == save_interval-1:
            if not args.debug == 1:
                print("Saving model...")
                torch.save(model.state_dict(), model_savepath)

    print("Finished epoch. Saving model...")
    if not args.debug == 1:
        torch.save(model.state_dict(), model_savepath)

print('Finished Training')
