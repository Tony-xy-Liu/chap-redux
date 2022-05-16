## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## 
## this is now a legacy file for archival purposes only!
## 
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

__author__ = "Abdurrahman Abul-Basher"
__date__ = '16/09/2019'
__copyright__ = "Copyright 2019, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman Abul-Basher"
__email__ = "arbasher@student.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform learning and prediction on dataset using CBT."

import datetime
import json
import os
import sys
import textwrap
from argparse import ArgumentParser

from .utility import file_path as fph
from .__train import train
from .utility.arguments import Arguments
from .model import *

def __print_header():
    if sys.platform.startswith('win'):
        os.system("cls")
    else:
        os.system("clear")
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__,
                                                "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45,
                                                 subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs
    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1
    arg.shuffle = parse_args.shuffle
    arg.subsample_input_size = parse_args.ssample_input_size
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.num_epochs = parse_args.num_epochs
    arg.batch = parse_args.batch

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.logpath = parse_args.logpath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.features_name = parse_args.features_name
    arg.X_name = parse_args.X_name
    arg.text_name = parse_args.text_name
    arg.M_name = parse_args.M_name
    arg.vocab_name = parse_args.vocab_name
    arg.file_name = parse_args.file_name
    arg.model_name = parse_args.model_name

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.train = parse_args.train
    arg.evaluate = parse_args.evaluate
    arg.transform = parse_args.transform
    arg.soap = parse_args.soap
    arg.spreat = parse_args.spreat
    arg.ctm = parse_args.ctm
    arg.lda = parse_args.lda
    arg.num_components = parse_args.num_components
    arg.alpha_mu = parse_args.alpha_mu
    arg.alpha_sigma = parse_args.alpha_sigma
    arg.alpha_phi = parse_args.alpha_phi
    arg.gamma = parse_args.gamma
    arg.kappa = parse_args.kappa
    arg.xi = parse_args.xi
    arg.varpi = parse_args.varpi
    arg.opt_method = parse_args.opt_method
    arg.cost_threshold = parse_args.cost_threshold
    arg.component_threshold = parse_args.component_threshold
    arg.forgetting_rate = parse_args.fr
    arg.delay_factor = parse_args.delay
    arg.top_k = parse_args.top_k
    arg.minimum_probability = parse_args.minimum_probability
    arg.collapse2ctm = parse_args.collapse2ctm
    arg.use_features = parse_args.use_features
    arg.use_supplement = parse_args.use_supplement
    arg.cal_average = parse_args.cal_average
    arg.max_sampling = parse_args.max_sampling

    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run CBT.")
    parser.add_argument('--display-interval', default=-1, type=int,
                        help='display intervals. -1 means display per each iteration. (default value: 2).')
    parser.add_argument('--random_state', default=12345,
                        type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=1,
                        help='Number of parallel workers. (default value: 1).')
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch size. (default value: 100).')
    parser.add_argument('--max-inner-iter', default=5, type=int,
                        help='Number of inner iteration inside a single epoch. '
                             'If batch = 1 better to set to 5. (default value: 5).')
    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of epochs over the training set. (default value: 10).')
    parser.add_argument('--ssample-input-size', default=0.1, type=float,
                        help='The size of input subsample. (default value: 0.1)')

    # Arguments for path
    parser.add_argument('--ospath', default=fph.OBJECT_PATH, type=str,
                        help='The path to the data object that contains extracted '
                             'information from the MetaCyc database. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--mdpath', default=fph.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--logpath', default=fph.LOG_PATH, type=str,
                        help='The path to the log directory.')

    # Arguments for file names and models
    parser.add_argument('--features-name', type=str, default='biocyc_features.pkl',
                        help='The features file name. (default value: "biocyc_features.pkl")')
    parser.add_argument('--X-name', type=str, default='biocyc_X.pkl',
                        help='The X file name. (default value: "biocyc_X.pkl")')
    parser.add_argument('--text-name', type=str, default='biocyc_text_X.pkl',
                        help='The file name to a list of strings. (default value: "biocyc_text_X.pkl")')
    parser.add_argument('--M-name', type=str, default='biocyc_M.pkl',
                        help='The M file name. (default value: "biocyc_M.pkl")')
    parser.add_argument('--vocab-name', type=str, default='biocyc_dictionary.pkl',
                        help='The vocab file name. (default value: "biocyc_dictionary.pkl")')
    parser.add_argument('--file-name', type=str, default='golden_9',
                        help='The file name to save an object. (default value: "golden_9")')
    parser.add_argument('--model-name', type=str, default='biocyc_9',
                        help='The file name, excluding extension, to save an object. (default value: "biocyc_9")')

    # Arguments for training and evaluation
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a model (SOAP, SPREAT, CTM, and LDA). '
                             '(default value: False).')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate models (SOAP, SPREAT, CTM, and LDA) performances. '
                             '(default value: False).')
    parser.add_argument('--transform', action='store_true', default=False,
                        help='Whether to transform concepts distribution from inputs using '
                             'a pretrained model (SOAP, SPREAT, CTM, and LDA). (default value: False).')
    parser.add_argument('--soap', action='store_true', default=False,
                        help='Whether to run SOAP. (default value: False).')
    parser.add_argument('--spreat', action='store_true', default=False,
                        help='Whether to run SPREAT. (default value: False).')
    parser.add_argument('--ctm', action='store_true', default=False,
                        help='Whether to run CTM. (default value: False).')
    parser.add_argument('--lda', action='store_true', default=False,
                        help='Whether to run LDA. (default value: False).')
    parser.add_argument("--num-components", type=int, default=200,
                        help="Total number of components. (default value: 200).")
    parser.add_argument("--alpha-mu", type=float, default=0.0001,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 0.0001).")
    parser.add_argument("--alpha-sigma", type=float, default=0.0001,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 0.0001).")
    parser.add_argument("--alpha-phi", type=float, default=0.0001,
                        help="hyper-parameter for Dirichlet distribution of feature. "
                             "(default value: 0). [1.0/number_of_types]")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 2).")
    parser.add_argument("--kappa", type=float, default=3.0,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 3).")
    parser.add_argument("--xi", type=float, default=0.0,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 0).")
    parser.add_argument("--varpi", type=float, default=1.0,
                        help="A hyper-parameter for logistic normal distribution of component. (default value: 0).")
    parser.add_argument("--opt-method", type=str, default="Newton-CG",
                        help="Optimization method for logistic normal distribution. (default value: L-BFGS-B).")
    parser.add_argument("--cost-threshold", type=float, default=0.001,
                        help="Break updates if the no relative change in the cost. (default value: 0.001).")
    parser.add_argument("--component-threshold", type=float, default=0.001,
                        help="Stopping tolerance for updating sample component distribution in E-step.")
    parser.add_argument('--fr', type=float, default=0.9, help='Forgetting rate to control how quickly old '
                                                              'information is forgotten. The value should '
                                                              'be set within the range of (0.5, 1.0] to guarantee asymptotic '
                                                              'convergence. (default value: 0.9).')
    parser.add_argument('--delay', type=float, default=1., help='A hyper-parameter to down weights early iterations.'
                                                                ' (default value: 1).')
    parser.add_argument('--collapse2ctm', action='store_true', default=False,
                        help='Whether to collapse SOAP/SPREAT model to CTM. (default value: False).')
    parser.add_argument('--use-features', action='store_true', default=False,
                        help='Whether to employ external component features. (default value: False).')
    parser.add_argument('--use-supplement', action='store_true', default=False,
                        help='Whether to add supplementary components for each sample. (default value: False).')
    parser.add_argument('--cal-average', action='store_true', default=False,
                        help='Whether to calculate the expected predictive distribution. (default value: False).')
    parser.add_argument('--max-sampling', default=3, type=int,
                        help='Maximum number of random samplings. (default value: 3).')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top k features (per component) to be considered for sparseness '
                             '(SOAP and SPREAT only). This hyperparameter is also used for computing '
                             'the coherence of top k features (for evaluation purposes). '
                             '(default value: 20).')
    parser.add_argument('--minimum-probability', type=float, default=0.0001,
                        help='Minimum probability to be considered for retrieving results. (default value: 0.0001).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data should be shuffled after each epoch. (default value: True).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    parse_command_line()
