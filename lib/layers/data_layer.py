import caffe
from config import cfg
import numpy as np
import yaml
import random
import scipy.misc

from multiprocessing import Process, Queue, Event

from utils.blob import im_list_to_blob, label_list_to_blob
from utils.timer import Timer


def get_minibatch(db, phase, batch_size, repeat):
    gen = db.get_set(phase.lower(), repeat=repeat)
    ims, labels = [], []

    for (imf, labelf) in gen:
        im = db.read_datum(imf)
        label = db.read_label(labelf)
        ims.append(im)
        labels.append(label)
        if len(ims) >= batch_size:
            break

    blobs = {'input': im_list_to_blob(ims),
             'label': label_list_to_blob(labels)}

    return blobs


class DataLayer(caffe.Layer):

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        self._timer.tic()
        if cfg.USE_PREFETCH:
            batch = self._blob_queue.get()
        else:
            batch = get_minibatch(self._db, self._phase, self._batch_size,
                                  self._repeat, self._is_flow)
        self._timer.toc()
        return batch

    @property
    def average_time(self):
        return self._timer.average_time

    def has_next(self):
        if cfg.USE_PREFETCH:
            raise ValueError('Cannot be used with blob prefetcher')

        return self._db.has_next(self._phase)

    def setup_db(self):
        # self._db = ExamplDB()
        raise NotImplementedError()

    def setup(self, bottom, top):
        """Setup the DataLayer."""

        # parse the layer parameter string, which must be valid YAML
        # Top layer mapping
        self._name_to_top_map = {'input': 0, 'label': 1}

        # TODO get layer params directly. No python param for phase
        layer_params = yaml.load(self.param_str)
        self._phase = layer_params['phase'].lower()
        self._batch_size = cfg.BATCH_SIZE
        self._repeat = layer_params.get('repeat', True)

        # Instantiate the class and implement the function
        self.setup_db()

        # Default shape
        in_shape = cfg.DEFAULT_INPUT_SHAPE
        label_shape = cfg.DEFAULT_LABEL_SHAPE

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._batch_size, in_shape*)
        top[1].reshape(self._batch_size, label_shape*)

        self._timer = Timer()

        if cfg.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._db,
                                                 layer_params)
            self._prefetch_process.start()

            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()

            import atexit
            atexit.register(cleanup)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        self._timer.tic()
        blobs = self._get_next_minibatch()
        self._timer.toc()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, db, layer_params):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        self._phase = layer_params['phase'].lower()
        self._repeat = layer_params.get('repeat', True)
        self._batch_size = cfg.BATCH_SIZE

        # Save the number of images
        self.exit = Event()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def shutdown(self):
        self.exit.set()

    def run(self):
        while not self.exit.is_set():
            blobs = get_minibatch(self._db, self._phase, self._batch_size,
                                  self._repeat, self._is_flow)
            self._queue.put(blobs)
