#!/usr/bin/env python
import _init_paths

import sys
import caffe
import argparse
import pprint
import numpy as np

from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from solver import Solver


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='a network prototxt file',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=4000, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='path to a caffemodel used to initialize the network with the weight',
                        default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='Batch size',
                        default=cfg.BATCH_SIZE, type=int)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--exp', dest='exp_dir',
                        help='experiment directory',
                        default=None, type=str)
    parser.add_argument('--no-prefetch', dest='no_prefetch',
                        help='Disable the parallel data prefetcher',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize initial network weights',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    pprint.pprint(args)

    # set the global cfg variable from the file
    if args.cfg_file is not None:
        for cfg_file in args.cfg_file:
            cfg_from_file(cfg_file)

    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.no_prefetch:
        cfg_from_list(['USE_PREFETCH', False])

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    output_dir = get_output_dir(None, args.exp_dir)
    print('Output will be saved to `{:s}`'.format(output_dir))

    if args.solver:
        # If solver is set, train the network using the solver prototxt.
        sw = Solver(args.solver, output_dir, args.weights)
        print('Training...')
        sw.train(args.max_iters)
        print('Finished Training')
    elif args.weights is not None and args.net is not None:
        # Run test
        net = caffe.Net(args.net, args.weights, caffe.TEST)
        raise NotImplementedError()
    else:
        print('Must provide either a caffemodel with a net definition or a solver prototxt')
