import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10, ImageNet, CelebA, Flowers102


def get_bernoulli_loader(batch_size, N, p, dim=1, train=True):
    '''
    Produces a DataLoader for a Bernoulli distribution
    with parameter p of size N.

    Args:
        batch_size (int): Batch size
        N (int): Number of samples (size of the dataset)
        p (float): Probability of success
        dim (int): Dimensionality of the data
        train (bool): Whether to use the training set
    Returns:
        DataLoader: DataLoader for the Bernoulli distribution
    '''
    data = torch.bernoulli(torch.full((N, dim), p))
    # torch.zeros(N) to make each sample (x, 0) for valid (label) embedding
    dataset = TensorDataset(data, torch.zeros(N))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_2D_gaussian_loader(batch_size, N, rho, train=True):
    '''
    Produces a DataLoader for a 2D Gaussian distribution
    with the covariance matrix [[1, rho], [rho, 1]] of size N.

    Args:
        batch_size (int): Batch size
        N (int): Number of samples (size of the dataset)
        rho (float): Correlation coefficient
        train (bool): Whether to use the training set

    Returns:
        DataLoader: DataLoader for the 2D Gaussian distribution
    '''
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, rho], [rho, 1.0]])
    dist = torch.distributions.MultivariateNormal(
        loc=mean, covariance_matrix=cov)
    data = dist.sample((N,))
    # torch.zeros(N) to make each sample ((x, y), 0) for valid (label) embedding
    dataset = TensorDataset(data, torch.zeros(N))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_mnist_loader(batch_size, train=True, SUPR=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = MNIST(
        root="./extra/data" if not SUPR else "/mimer/NOBACKUP/Datasets/MNIST/raw",
        train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_mnist_loader_digit(batch_size, digit, train=True, SUPR=False):
    assert digit in range(10)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = MNIST(
        root="./extra/data" if not SUPR else "/mimer/NOBACKUP/Datasets/MNIST/raw",
        train=train, download=True, transform=transform
    )

    # 0 to make valid embedding when `n_classes=1`
    data = [(img, 0) for img, label in dataset if label == digit]

    return DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)


def get_cifar10_loader(batch_size, train=True, SUPR=False):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CIFAR10(
        root="./extra/data" if not SUPR else "/mimer/NOBACKUP/Datasets/CIFAR",
        train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_cifar10_dogs_loader(batch_size, train=True, SUPR=False):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CIFAR10(
        root="./extra/data" if not SUPR else "/mimer/NOBACKUP/Datasets/CIFAR",
        train=train, download=True, transform=transform
    )

    # 0 to make valid embedding when `n_classes=1`
    data = [(img, 0) for img, label in dataset if label == 5]

    return DataLoader(data, batch_size=batch_size, shuffle=True)


def get_cifar10_cats_loader(batch_size, train=True, SUPR=False):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CIFAR10(
        root="./extra/data" if not SUPR else "/mimer/NOBACKUP/Datasets/CIFAR",
        train=train, download=True, transform=transform
    )

    # 1 to make valid embedding when `n_classes=2`
    data = [(img, 1) for img, label in dataset if label == 3]

    return DataLoader(data, batch_size=batch_size, shuffle=True)


def get_imagenet_loader(img_size, batch_size, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageNet(
        root="/mimer/NOBACKUP/Datasets/ImageNet",
        split="train" if train else "val",
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_celeba_loader(img_size, batch_size, train=True):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = CelebA(
        root="./extra/data",
        split="train" if train else "test",
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_oxford_flowers_loader(img_size, batch_size, train=True):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size) if train else transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = Flowers102(
        root="./extra/data",
        split="train" if train else "test",
        download=True,
        transform=transform,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_oxford_flowers_loader_class(img_size, batch_size, class_idx, train=True):
    """
    Get a dataloader for a specific class from the Oxford Flowers 102 dataset.

    Args:
        batch_size (int): Batch size
        class_idx (int): Class index (0-101)
        train (bool): Whether to use the training set
        SUPR (bool): Whether we're running on SUPR cluster

    Returns:
        DataLoader: DataLoader for the specified class
    """
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size) if train else transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = Flowers102(
        root="./extra/data",
        split="train" if train else "test",
        download=True,
        transform=transform,
    )

    dataset = [(img, 0) for img, label in dataset if label == class_idx]

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True)


def get_flowers_two_class_loader(img_size, batch_size, class1, class2, train=True):
    """
    Get dataloaders for two specific classes from the Oxford Flowers 102 dataset.
    Useful for two-sided flow matching.

    Args:
        batch_size (int): Batch size
        class1 (int): First class index (0-101)
        class2 (int): Second class index (0-101)
        train (bool): Whether to use the training set
        SUPR (bool): Whether we're running on SUPR cluster

    Returns:
        (DataLoader, DataLoader): DataLoaders for the two specified classes
    """

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size) if train else transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = Flowers102(
        root="./extra/data",
        split="train" if train else "test",
        download=True,
        transform=transform,
    )

    class1_data = [(img, 0) for img, label in dataset if label == class1]
    class2_data = [(img, 1) for img, label in dataset if label == class2]

    loader1 = DataLoader(class1_data, batch_size=batch_size,
                         shuffle=train, drop_last=True)
    loader2 = DataLoader(class2_data, batch_size=batch_size,
                         shuffle=train, drop_last=True)

    return loader1, loader2


def get_data_from_path(args):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
