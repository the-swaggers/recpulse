import struct
import os
import gzip
import recpulse_cuda as rp

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


def load_mnist(data_dir='data/mnist', download=True):
    if download and HAS_URLLIB:
        _download_mnist(data_dir)

    train_images = _load_idx_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = _load_idx_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = _load_idx_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = _load_idx_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    return train_images, train_labels, test_images, test_labels


def _download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for f in files:
        path = os.path.join(data_dir, f)
        unpacked = path[:-3]
        if os.path.exists(unpacked):
            continue
        if not os.path.exists(path):
            print(f'Downloading {f}...')
            urllib.request.urlretrieve(base_url + f, path)
        print(f'Extracting {f}...')
        with gzip.open(path, 'rb') as gz:
            with open(unpacked, 'wb') as out:
                out.write(gz.read())


def _load_idx_images(path):
    import numpy as np

    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} for images')
        data = f.read()

    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32).reshape(num, rows * cols) / 255.0
    return rp.from_numpy(arr)


def _load_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} for labels')
        data = f.read()

    return list(data)


def get_batch(images, labels, batch_idx, batch_size):
    start = batch_idx * batch_size
    end = min(start + batch_size, images.shape[0])
    if start >= images.shape[0]:
        return None, None

    batch_images = images.slice([start, 0], [end, images.shape[1]], [1, 1])
    batch_labels = labels[start:end]
    return batch_images, batch_labels
