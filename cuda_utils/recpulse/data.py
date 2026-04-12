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


def load_cifar10(data_dir='data/cifar10', download=True):
    import numpy as np
    import pickle
    import tarfile

    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    if download and HAS_URLLIB and not os.path.exists(batch_dir):
        os.makedirs(data_dir, exist_ok=True)
        tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
        if not os.path.exists(tar_path):
            print('Downloading CIFAR-10...')
            urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', tar_path)
        print('Extracting CIFAR-10...')
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)

    def load_batch(path):
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'], d[b'labels']

    train_x, train_y = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(batch_dir, f'data_batch_{i}'))
        train_x.append(x)
        train_y.extend(y)

    train_x = np.concatenate(train_x).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_x, test_y = load_batch(os.path.join(batch_dir, 'test_batch'))
    test_x = np.array(test_x).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    train_images = rp.from_numpy(train_x)
    test_images = rp.from_numpy(test_x)

    return train_images, train_y, test_images, test_y


def get_batch_4d(images, labels, batch_idx, batch_size):
    start = batch_idx * batch_size
    end = min(start + batch_size, images.shape[0])
    if start >= images.shape[0]:
        return None, None

    batch_images = images.slice([start, 0, 0, 0],
                                [end, images.shape[1], images.shape[2], images.shape[3]],
                                [1, 1, 1, 1])
    batch_labels = labels[start:end]
    return batch_images, batch_labels


def get_batch(images, labels, batch_idx, batch_size):
    start = batch_idx * batch_size
    end = min(start + batch_size, images.shape[0])
    if start >= images.shape[0]:
        return None, None

    batch_images = images.slice([start, 0], [end, images.shape[1]], [1, 1])
    batch_labels = labels[start:end]
    return batch_images, batch_labels
