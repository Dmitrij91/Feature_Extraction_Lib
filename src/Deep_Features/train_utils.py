
import json

def Prepare_Training_Instance(Model_hash = ""):
    config_path = '../src/Deep_Features/training_config.json'
    if not os.path.isfile(config_path):
        print("Please provide training config.")
        exit()
    with open(config_path, "r") as config_file:
        config = json.loads(config_file.read())
    if not Model_hash == 0:
        training_hash = Model_hash
    else:
        import hashlib
        import time
        training_hash = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    model_identifier = f"{config['model']}{config['model_size']}_{Model_hash}"
    model_savedir = f"../src/Deep_Features/results/trained_models/{model_identifier}"
    # If directory empty create new directory
    if (not os.path.isdir(model_savedir)):
        os.mkdir(model_savedir)
    model_savepath = os.path.join(model_savedir, f"{model_identifier}.pt")
    # Load directory for saving running data for tensorboard 
    summary_logdir = f"../src/Deep_Features//runs/{model_identifier}"
    if (not os.path.isdir(summary_logdir)):
        os.mkdir(summary_logdir)
    # copy config file to places one might refer to
    copyfile(config_path, os.path.join(model_savedir, "training_config.json"))
    copyfile(config_path, os.path.join(summary_logdir, "training_config.json"))
    return model_savepath,summary_logdir
