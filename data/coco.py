import torch
import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform, train_aug_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers,noise,noiseType,noiseLevel):
    """
    Loading nus-wide dataset.

    Args:
        tc(int): Top class.
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    Coco.init(root, num_query, num_train)
    query_dataset = Coco(root, 'query', query_transform())
    train_dataset = Coco(root, 'train', train_transform(), \
        transform_aug = train_aug_transform(),noise = noise, noise_type=noiseType, closeset_ratio=noiseLevel)
    retrieval_dataset = Coco(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader



class Coco(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None, transform_aug=None,\
        noise = True, noise_type='pairflip',closeset_ratio=0, openset_ratio=0, random_state=0, verbose=True):
        self.root = root
        self.transform = transform
        self.transform_aug= transform_aug
        self.mode = mode

        if mode == 'train':
            if noise:
                self.data = Coco.TRAIN_DATA
                self.targets = Coco.TRAIN_TARGETS
                self.origin_targets = Coco.TRAIN_TARGETS
                train_labels = np.asarray([self.targets[i] for i in range(len(self.targets))])
                self.targets, self.actual_noise_rate = noisify_dataset(80, train_labels, noise_type, closeset_ratio,\
                                                                   openset_ratio, random_state, verbose)
            else:
                self.data = Coco.TRAIN_DATA
                self.targets = Coco.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Coco.QUERY_DATA
            self.targets = Coco.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Coco.RETRIEVAL_DATA
            self.targets = Coco.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img_ = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img_)
            
        
        if self.mode == 'train':
            if self.transform_aug is not None:
                img_aug = self.transform_aug(img_)
            else: 
                img_aug = self.transform_aug(img_)
            
            return img, img_aug, self.targets[index], index
        else:
            return img, self.targets[index], index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        """
        Initialize dataset

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        img_txt_path = os.path.join(root, 'database.txt')
        

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([val.split()[0].replace('/home/caozhangjie/run-czj/dataset/coco/','',1) for val in f])
        with open(img_txt_path, 'r') as f:
            targets = np.array([np.array([int(la) for la in val.split()[1:]]) for val in f])


        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        Coco.QUERY_DATA = data[query_index]
        Coco.QUERY_TARGETS = targets[query_index, :]

        Coco.TRAIN_DATA = data[train_index]
        Coco.TRAIN_TARGETS = targets[train_index, :]

        Coco.RETRIEVAL_DATA = data[retrieval_index]
        Coco.RETRIEVAL_TARGETS = targets[retrieval_index, :]

# basic function
# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio=0.8, nb_classes=10):
    """

    Example of the noise transition matrix (closeset_ratio = 0.3):
        - Symmetric:
            -                               -
            | 0.7  0.1  0.1  0.1  0.0  0.0  |
            | 0.1  0.7  0.1  0.1  0.0  0.0  |
            | 0.1  0.1  0.7  0.1  0.0  0.0  |
            | 0.1  0.1  0.1  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -
        - Pairflip
            -                               -
            | 0.7  0.3  0.0  0.0  0.0  0.0  |
            | 0.0  0.7  0.3  0.0  0.0  0.0  |
            | 0.0  0.0  0.7  0.3  0.0  0.0  |
            | 0.3  0.0  0.0  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -

    """
    assert closeset_noise_ratio > 0.0, 'noise rate must be greater than 0.0'
    assert 0.0 <= openset_noise_ratio < 1.0, 'the ratio of out-of-distribution class must be within [0.0, 1.0)'
    closeset_nb_classes = int(nb_classes * (1 - openset_noise_ratio))
    # openset_nb_classes = nb_classes - closeset_nb_classes
    if noise_type == 'symmetric':
        P = np.ones((nb_classes, nb_classes))
        P = (closeset_noise_ratio / (closeset_nb_classes - 1)) * P
        for i in range(closeset_nb_classes):
            P[i, i] = 1.0 - closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    elif noise_type == 'pairflip':
        P = np.eye(nb_classes)
        P[0, 0], P[0, 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        for i in range(1, closeset_nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        P[closeset_nb_classes - 1, closeset_nb_classes - 1] = 1.0 - closeset_noise_ratio
        P[closeset_nb_classes - 1, 0] = closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    else:
        raise AssertionError("noise type must be either symmetric or pairflip")
    return P


def noisify(y_train, noise_transition_matrix, random_state=None):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    return y_train_noisy, actual_noise


def noisify_dataset(nb_classes=24, train_labels=None, noise_type=None,
                    closeset_noise_ratio=0.0, openset_noise_ratio=0.0, random_state=0, verbose=True):
    noise_transition_matrix = generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio, nb_classes)
    train_noisy_labels, actual_noise_rate = noisify(train_labels, noise_transition_matrix, random_state)
    # if verbose:
    #     print(f'Noise Transition Matrix: \n {noise_transition_matrix}')
    #     print(f'Noise Type: {noise_type} (close set: {closeset_noise_ratio}, open set: {openset_noise_ratio})\n'
    #           f'Actual Total Noise Ratio: {actual_noise_rate:.3f}')
    return train_noisy_labels, actual_noise_rate



 