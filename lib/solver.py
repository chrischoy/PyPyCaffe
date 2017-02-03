"""Train semantic segmentation network."""
import re
import os
import math
import json
import numpy as np
from matplotlib import gridspec
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2

from config import cfg
from utils.timer import Timer


class Solver(object):

    def __init__(self, solver_prototxt, output_dir, pretrained_model=None):
        """
        Initialize the SolverWrapper.
        dataset must be a imdb type
        """

        self.output_dir = output_dir

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

        if self.solver_param.solver_type == \
                caffe_pb2.SolverParameter.SolverType.Value('SGD'):
            self.solver = caffe.SGDSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
            caffe_pb2.SolverParameter.SolverType.Value('NESTEROV'):
            self.solver = caffe.NesterovSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
            caffe_pb2.SolverParameter.SolverType.Value('ADAGRAD'):
            self.solver = caffe.AdaGradSolver(solver_prototxt)
        elif self.solver_param.solver_type == \
            caffe_pb2.SolverParameter.SolverType.Value('ADAM'):
            self.solver = caffe.AdamSolver(solver_prototxt)
        else:
            raise NotImplementedError('Solver type not defined')

        if pretrained_model is not None:
            print('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

    def check(self):
        print('Iteration', self.solver.iter, '----------------')
        kill = False
        for key, param in self.solver.net.params.iteritems():
            blob_max = np.max(np.abs(self.solver.net.blobs[key].data))
            param_max = np.max(np.abs(param[0].data))
            bias_max = np.max(np.abs(param[1].data))
            diff_max = np.max(param[0].diff)
            diff_min = np.min(param[0].diff)
            bias_diff_max = np.max(param[1].diff)
            bias_diff_min = np.min(param[1].diff)
            if math.isnan(param_max) or math.isnan(diff_max):
                kill = True
            print(key, 'blob:', blob_max,
                  'weight:', param_max,
                  'bias', bias_max,
                  'diff:[', diff_min, diff_max, '] ',
                  'bias_diff: [', bias_diff_min, bias_diff_max, ']')

        print('loss', self.solver.loss)
        return kill

    def train(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()

        self.loss = np.zeros((max_iters,))
        iter_ = 0

        # For debugging purpose
        kill = False

        # Check whether the network is initialized
        self.check()

        # Unlike self.solver.solve(), this allows exit
        while self.solver.iter < max_iters:

            timer.tic()
            self.solver.step(1)
            timer.toc()

            if self.solver.iter % 400 == 0 or self.solver.iter == 1:
                if self.check():
                    return

            self.loss[iter_] = self.solver.loss

            if self.solver.loss > cfg.TRAIN.LOSS_THRESHOLD or kill:
                print('Loss: {} > Thresh {} or NAN detected. Exit'.format(
                    self.solver.loss, cfg.TRAIN.LOSS_THRESHOLD))
                return

            if self.solver.iter % self.solver_param.display == 0:
                print('Total training speed: {:.3f}s / iter'.format(timer.average_time))
                # Print python layer processing speed
                for layer in self.solver.net.layers:
                    if hasattr(layer, 'average_time'):
                        print('%s: %.3gs / iter' % (layer, layer.average_time))

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

            iter_ += 1

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

    def snapshot(self):
        """
        Take a snapshot of the network
        """
        net = self.solver.net

        # Save the network parameters
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))

        # Save the loss history into a json file
        filename = (self.solver_param.snapshot_prefix +
                    '_loss_iter_{:d}'.format(self.solver.iter) + '.json')
        filename = os.path.join(self.output_dir, filename)
        with open(filename,'w') as f:
            str_loss = json.dumps(self.loss.tolist())
            f.write(str_loss)

        print('Saved the current snapshot to: {:s}'.format(filename))
