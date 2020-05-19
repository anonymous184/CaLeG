import argparse
import logging
import os

import configs
import utils
from datahandler import DataHandler

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["OMP_NUM_THREADS"] = "8"

parser = argparse.ArgumentParser()
parser.add_argument("--imbalance", default=True, help='whether to use imbalanced data')
parser.add_argument("--aug-rate", type=float, default=1.0, help='sampling rate r')
parser.add_argument("--dataset", type=str, default='mnist',
                    help='dataset name', choices=('mnist', 'fashion', 'cifar10', 'svhn'))
FLAGS = parser.parse_args()
if FLAGS.aug_rate > 0.0:
    from caleg import CaLeG
else:
    from caleg_noaug import CaLeG


def main(exp, tag, seed):
    if exp == 'mnist':
        opts = configs.config_mnist
    elif exp == 'fashion':
        opts = configs.config_fashion
    elif exp == 'svhn':
        opts = configs.config_SVHN
    elif exp == 'cifar10':
        opts = configs.config_cifar10
    else:
        assert False, 'Unknown experiment configuration'

    opts['imbalance'] = FLAGS.imbalance
    opts['work_dir'] = data_dir
    opts['aug_rate'] = FLAGS.aug_rate
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'], 'checkpoints'))

    # Dumping all the configs to the text file
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset
    data = DataHandler(opts, seed)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    model = CaLeG(opts, tag)
    model.train(data)
    del model


if __name__ == '__main__':
    for seed in range(1, 6):
        tag = '%s_seed%02d' % (FLAGS.dataset, seed)
        data_dir = './results/%s/' % tag
        main(FLAGS.dataset, tag, seed)
