import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import random 
from shutil import copyfile
from sys import exit

def Prepare_Training_Instance(Model_hash = ""):
    #cur_path = os.path.abspath(os.getcwd())
    config_path = 'training_config.json'
    if not os.path.isfile(config_path):
        print("Please provide training config.")
        exit
    with open(config_path, "r") as config_file:
        config = json.loads(config_file.read())
    if not Model_hash == "":
        training_hash = Model_hash
    else:
        import hashlib
        import time
        training_hash = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    model_identifier = f"{config['model']}{config['model_size']}_{training_hash}"
    model_savedir = f"./results/trained_models/{model_identifier}"
    # If directory empty create new directory
    if (not os.path.isdir(model_savedir)):
        os.mkdir(model_savedir)
    model_savepath = os.path.join(model_savedir, f"{model_identifier}.pt")
    # Load directory for saving running data for tensorboard 
    summary_logdir = f"./runs/{model_identifier}"
    if (not os.path.isdir(summary_logdir)):
        os.mkdir(summary_logdir)
    # copy config file to places one might refer to
    copyfile(config_path, os.path.join(model_savedir, "training_config.json"))
    copyfile(config_path, os.path.join(summary_logdir, "training_config.json"))
    return model_savepath,summary_logdir

def generate_paths(num, base_path):
    num_str = f"{num:03}"  # Format with zero padding
    path_1 = f"{base_path}/horse/horse{num_str}.png"
    path_2 = f"{base_path}/mask/horse{num_str}.png"
    return path_1, path_2

def save_dataset(train_indices, test_indices, base_path, train_file, test_file):
    Train_Paths, Train_Paths_Labels = zip(*[generate_paths(num, base_path) for num in train_indices])
    Test_Paths, Test_Paths_Labels = zip(*[generate_paths(num, base_path) for num in test_indices])

    D_Train = pd.DataFrame({'Train_Path': Train_Paths, 'Train_Path_Labels': Train_Paths_Labels})
    D_Test = pd.DataFrame({'Test_Path': Test_Paths, 'Test_Path_Labels': Test_Paths_Labels})

    D_Train.to_csv(train_file)
    D_Test.to_csv(test_file)

def load_or_generate_data(base_path, train_file, test_file, total_images, test_size=0.9, num_indices = 10, resume=False):
    if resume and os.path.exists(train_file) and os.path.exists(test_file):
        D_Train = pd.read_csv(train_file)
        D_Test = pd.read_csv(test_file)
    else:
        indices = random.sample(range(1, total_images), num_indices)
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        save_dataset(train_indices, test_indices, base_path, train_file, test_file)
        D_Train = pd.read_csv(train_file)
        D_Test = pd.read_csv(test_file)
    return D_Train, D_Test
