from collections import OrderedDict
import hashlib, os, pickle
from util.colors import BOLD, ORANGE, ENDC

# Logger stores in trained_models by default
def create_logger(args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__
    assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
    assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
    assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    seed = str(arg_dict.pop("seed"))
    logdir = str(arg_dict.pop('logdir'))
    env_name = str(arg_dict['env_name'])

    # see if this run has a unique name, if so then that is going to be the name of the folder
    if run_name is not None:
        logdir = os.path.join(logdir, env_name)
        output_dir = os.path.join(logdir, run_name)
        # Check if policy name already exists. If it does, increment filename
        index = ''
        while os.path.exists(output_dir + index):
            if index:
                index = '_(' + str(int(index[2:-1]) + 1) + ')'
            else:
                index = '_(1)'
        output_dir += index
    else:
        # see if we are resuming a previous run, if we are mark as continued
        if hasattr(args, 'previous') and args.previous is not None:
            if args.exchange_reward is not None:
                output_dir = args.previous[0:-1] + "_NEW-" + args.reward
            else:
                print(args.previous[0:-1])
                output_dir = args.previous[0:-1] + '-cont'
        else:
            # get a unique hash for the hyperparameter settings, truncated at 10 chars
            arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
            logdir     = os.path.join(logdir, env_name)
            output_dir = os.path.join(logdir, arg_hash)

    # create a directory with the hyperparm hash as its name, if it doesn't
    # already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    logger = SummaryWriter(output_dir, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + BOLD + ORANGE + str(output_dir) + ENDC)

    logger.dir = output_dir
    return logger
