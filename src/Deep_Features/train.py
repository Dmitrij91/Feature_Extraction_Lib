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
from train_utils import Prepare_Training_Instance, load_or_generate_data

from dataloader_2d import build_data_loader

parser = argparse.ArgumentParser(description='Train neuronal network classifier for deep feature extraction.')
parser.add_argument("--resume_hash",
    type=str,
    help="Hash of previous training instance.",
    default="")
parser.add_argument("--debug",
    type=int,
    help="Run training in debug mode. No logs are created and limited data is loaded.",
    default=0)
parser.add_argument('--Cl_num',
    type = int , 
    help = " Number of classes of the underlying dataset  ",
    default = 2)
args = parser.parse_args()

config_path = 'training_config.json'
if not os.path.isfile(config_path):
    print('Enter')
    print("Please provide training config.")
    exit()
with open(config_path, "r") as config_file:
    config = json.loads(config_file.read())
    print(f" Train Classification Model with the configurations: \n{config}")

c = args.Cl_num

paths = ['./results/trained_models','./runs']

for folder_path in paths:
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# hash to identify this training instance with
if not len(args.resume_hash) == 0:
    training_hash = args.resume_hash
else:
    import hashlib
    import time
    training_hash = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())

size_window = (config['window_size']['y'], config['window_size']['x'])
device = torch.device('cpu')
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Constants
BASE_PATH = '../../../FM_Eikonal/data/weizmann_horse_db'
TRAIN_FILE = 'training_data.csv'
TEST_FILE = 'testing_data.csv'
TOTAL_IMAGES = 327

# Load Test and Train Data 

model_savepath,summary_logdir = Prepare_Training_Instance()
D_Train, D_Test = load_or_generate_data(BASE_PATH, TRAIN_FILE, TEST_FILE, TOTAL_IMAGES,test_size = 0.7,num_indices=TOTAL_IMAGES-1, resume=False)
# build model
if config['model'] == "resnet":
    from model_resnet import resnet
    model = resnet(config['model_size'], 1, c, dropout=config['dropout'],Feat=False)
elif config['model'] == "linear":
    from model_linear import linear_predictor
    model = linear_predictor(int(np.prod(size_window)), c)
elif config['model'] == "dense":
    from model_dense import dense_predictor
    model = dense_predictor(int(np.prod(size_window)), c, size=config['model_size'], activation=config['activation'], dropout=config['dropout'])
elif config['model'] == "flat_dense":
    from model_dense import flat_dense_predictor
    model = flat_dense_predictor(int(np.prod(size_window)), c, size=config['model_size'], activation=config['activation'], dropout=config['dropout'])
else:
    print(f"Unknown model {config['model']}.")
    exit()


path = os.path.split(os.getcwd())[0]+'/Deep_Features'
# Train the Model  

c = 2
model = resnet(config['model_size'], 1, c, dropout=config['dropout'],Hdim = 3,Feat=False)
fully_convolutional = (config['model'] in ['resnet'])
model.to(device)

train_dataloader = build_data_loader(size_window, c,path_dir=path,
    train=True,
    batch_size=config['batch_size'],
    workers=config['workers_training'],
    single_pixel_out=(fully_convolutional))

test_dataloader = build_data_loader(
    size_window, c,path_dir=path,
    train=False,
    batch_size=config['batch_size'],
    workers=config['workers_testing'],
    single_pixel_out=(fully_convolutional))
test_iter = iter(test_dataloader)




if os.path.isfile(model_savepath):
    model.load_state_dict(torch.load(model_savepath))

writer = SummaryWriter(summary_logdir)

# Train the Model  
criterion = nn.CrossEntropyLoss()
if config["optimizer"] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
elif config["optimizer"] == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=0.01)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Number of trainable network parameters is {params}")


log_interval = int(1e3)
save_interval = int(1e3)
num_test_batches = 20

running_loss = 0.0

for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        labels = torch.squeeze(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % log_interval == log_interval-1:
            writer.add_scalar('training_loss', running_loss / log_interval, epoch*len(train_dataloader) + i)
            print(f"batch {i} training loss {running_loss / log_interval}")
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
                test_labels = torch.squeeze(test_labels)
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)
                test_out = model(test_data)
                test_out = torch.squeeze(test_out, 1)
                test_loss += criterion(test_out, test_labels).item()
            print(f"batch {i} testing loss {test_loss / num_test_batches}")
            model.train()
        if i % save_interval == save_interval-1:
            print("Saving model...")
            torch.save(model.state_dict(), model_savepath)

    print("Finished epoch. Saving model...")
    torch.save(model.state_dict(), model_savepath)

    print("Finished epoch. Saving model...")
print('Finished Training')
