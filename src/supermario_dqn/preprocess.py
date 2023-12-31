"""
Utilities for frames preprocessing
"""

from torchvision import transforms

__all__ = ['preprocess']


# constants

_mario_dress = 240  # , 56, 0]
_mario_skin = 252  # , 16, 68]
_mario_other = 172  # , 140, 0]


def _preprocess0(tensor, resize_h, resize_w):
    state_ = tensor[70:210, :, 0]

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor()
    ])(state_)[0]


def _preprocess1(tensor, resize_h, resize_w):
    """
    levels with light blue backgrounds and red ground
    """
    background = 104

    state_ = tensor[70:210, :, 0]  # crop and red signal
    state_[state_ == _mario_dress] = 255
    state_[state_ == _mario_skin] = 255
    state_[state_ == _mario_other] = 255
    state_[state_ == background] = 0

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor()
    ])(state_)[0]


def _preprocess2(tensor, resize_h, resize_w):
    """
    levels with black background and blue ground
    """

    state_ = tensor[70:210, :, 0] + tensor[70:210, :, 2]
    state_[state_ == _mario_dress] = 255
    state_[state_ == _mario_skin] = 255
    state_[state_ == _mario_other] = 255

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor()
    ])(state_)[0]


def _preprocess3(tensor, resize_h, resize_w):
    """
    levels with black background and grey ground or sea
    """
    state_ = tensor[70:210, :, 0]  # crop and red signal
    state_[state_ == _mario_dress] = 255
    state_[state_ == _mario_skin] = 255
    state_[state_ == _mario_other] = 255

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor()
    ])(state_)[0]


_preprocess_map = {
    (1, 1): _preprocess1,
    (1, 3): _preprocess1,
    (2, 1): _preprocess1,
    (2, 3): _preprocess1,
    (4, 1): _preprocess1,
    (4, 3): _preprocess1,
    (5, 1): _preprocess1,
    (5, 2): _preprocess1,
    (5, 3): _preprocess1,
    (7, 1): _preprocess1,
    (7, 3): _preprocess1,
    (8, 1): _preprocess1,
    (8, 2): _preprocess1,
    (8, 3): _preprocess1,
    (1, 2): _preprocess2,
    (4, 2): _preprocess2,
    (1, 4): _preprocess3,
    (2, 4): _preprocess3,
    (3, 4): _preprocess3,
    (4, 4): _preprocess3,
    (5, 4): _preprocess3,
    (6, 4): _preprocess3,
    (7, 4): _preprocess3,
    (8, 4): _preprocess3,
    (6, 3): _preprocess3,
    (2, 2): _preprocess3,
    (7, 2): _preprocess3,
    (3, 1): _preprocess3,
    (3, 2): _preprocess3,
    (3, 3): _preprocess3,
    (6, 1): _preprocess3,
    (6, 2): _preprocess3,
}


def preprocess(world, stage, tensor, resize_h, resize_w):
    """
    Preprocess given level frame
    """

    if world not in range(1, 9):
        raise ValueError(f'world in 1..8, not {world}')
    if stage not in range(1, 5):
        raise ValueError(f'world in 1..4, not {stage}')

    return _preprocess_map[(world, stage)](tensor, resize_h, resize_w)
