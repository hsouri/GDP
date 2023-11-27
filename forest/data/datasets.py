"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision

from ..consts import *   # import all mean/std constants

import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import csv

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive

# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_datasets(dataset, data_path, normalize=True):
    """Construct datasets with appropriate transforms."""
    # Compute mean, std:
    if dataset == 'CIFAR100':
        trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
    elif dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    elif dataset == 'GTSRB':
        trainset = GTSRB(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if GTSRB_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = GTSRB_mean, GTSRB_std
    elif dataset == 'MNIST':
        trainset = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if mnist_mean is None:
            cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
            data_mean = (torch.mean(cc, dim=0).item(),)
            data_std = (torch.std(cc, dim=0).item(),)
        else:
            data_mean, data_std = mnist_mean, mnist_std
    elif dataset == 'ImageNet':
        trainset = ImageNet(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'ImageNet1k':
        trainset = ImageNet1k(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'TinyImageNet':
        trainset = TinyImageNet(root=data_path, split='train', transform=transforms.ToTensor())
        if tiny_imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = tiny_imagenet_mean, tiny_imagenet_std
    else:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if normalize:
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        trainset.data_mean = data_mean
        trainset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        trainset.data_std = (1.0, 1.0, 1.0)

    # Setup data
    if dataset in ['ImageNet', 'ImageNet1k']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    trainset.transform = transform_train

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    if dataset == 'CIFAR100':
        validset = CIFAR100(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10':
        validset = CIFAR10(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'MNIST':
        validset = MNIST(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'GTSRB':
        validset = GTSRB(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'TinyImageNet':
        validset = TinyImageNet(root=data_path, split='val', transform=transform_valid)
    elif dataset == 'ImageNet':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet(root=data_path, split='val', download=False, transform=transform_valid)
    elif dataset == 'ImageNet1k':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet1k(root=data_path, split='val', download=False, transform=transform_valid)

    if normalize:
        validset.data_mean = data_mean
        validset.data_std = data_std
    else:
        validset.data_mean = (0.0, 0.0, 0.0)
        validset.data_std = (1.0, 1.0, 1.0)

    return trainset, validset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


class Deltaset(torch.utils.data.Dataset):
    """Save delta in separate dataset."""

    def __init__(self, dataset, delta):
        """Save dataset and modification in a central location."""
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        """Return data + delta."""
        (img, target, index) = self.dataset[idx]
        return (img + self.delta[idx], target, index)

    def __len__(self):
        """Length is always datset length."""
        return len(self.dataset)


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index




class GTSRB(torchvision.datasets.VisionDataset):
    """German Traffic Sign benchmark."""

    base_folder = "GTSRB"
    url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    filenames = [
        "GTSRB_Final_Training_Images.zip",
        "GTSRB_Final_Test_Images.zip",
        "GTSRB_Final_Test_GT.zip"
    ]
    classes = ['20', '30', '50', '60', '70', '80', '80 aufgehoben', '100', '120', 'Überholverbot', 'LKW Überholverbot',
               'Vorfahrt', 'Vorfahrtstraße', 'Vorfahrt Achten', 'Stopp', 'Verboten', 'LKW Verboten', 'Einfahrt Verboten',
               'Achtung', 'Linkskurve', 'Rechtskurve', 'Doppelkurve', 'Unebene Fahrbahn', 'Schleudergefahr', 'Verengte Fahrbahn',
               'Bauarbeiten', 'Ampel', 'Fußgänger', 'Kinder', 'Fahrradfahrer', 'Schneeglätte', 'Wildtiere kreuzen',
               'Begrenzung aufgehoben', 'Rechts abbiegen', 'Links abbiegen', 'Geradeaus', 'Geradeaus oder Rechts',
               'Geradeaus oder Links', 'Rechts vorbei', 'Links vorbei', 'Kreisverkehr', 'Überholverbot aufgehoben',
               'LKW Überholverbot aufgehoben']

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, img_size=(32, 32)):
        """Initialize like this was CIFAR. We fix the image size for simplicity."""
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.img_size = img_size

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if train:
            self.file_paths = []
            self.labels = []
            folder_structure = os.path.join(self.root, self.base_folder, 'Final_Training', 'Images')
            for idx, folder in enumerate(sorted(os.listdir(folder_structure))):
                folder_path = os.path.join(folder_structure, folder)
                paths_for_this_label = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.ppm')]
                self.file_paths += paths_for_this_label
                self.labels += [idx for path in paths_for_this_label]
        else:
            folder_path = os.path.join(self.root, self.base_folder, 'Final_Test', 'Images')
            self.file_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.ppm')]
            self.labels = []
            with open(os.path.join(self.root, self.base_folder, 'GT-final_test.csv')) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                for row in reader:
                    self.labels += [int(row['ClassId'])]
            assert len(self.file_paths) == len(self.labels)

    def _check_integrity(self) -> bool:
        for fentry in self.filenames:
            if not os.path.exists(os.path.join(self.root, fentry)):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        for file in self.filenames:
            print('now looking for ' + os.path.join(self.url, file))
            download_and_extract_archive(os.path.join(self.url, file), self.root, filename=file, md5=None)
        os.rename(os.path.join(self.root, 'GT-final_test.csv'),
                  os.path.join(self.root, self.base_folder, 'GT-final_test.csv'))

    def __len__(self):
        """Length from actual file paths on system."""
        return len(self.file_paths)


    def __getitem__(self, index):
        """Getitem with index -> img, label, index."""
        fpath, label = self.file_paths[index], self.labels[index]

        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB").resize(self.img_size, resample=Image.BICUBIC)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, label, index

    def get_target(self, index):
        """index -> target, index."""
        label = self.labels[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return label, index


class ImageNet(torchvision.datasets.ImageNet):
    """Overwrite torchvision ImageNet to change metafile location if metafile cannot be written due to some reason."""

    def __init__(self, root, split='train', download=False, **kwargs):
        """Use as torchvision.datasets.ImageNet."""
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except RuntimeError:
            torchvision.datasets.imagenet.META_FILE = os.path.join(os.path.expanduser('~/data/'), 'meta.bin')
            try:
                wnid_to_classes = load_meta_file(self.root)[0]
            except RuntimeError:
                self.parse_archives()
                wnid_to_classes = load_meta_file(self.root)[0]

        torchvision.datasets.ImageFolder.__init__(self, self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        """Scrub class names to be a single string."""
        scrubbed_names = []
        for name in self.classes:
            if isinstance(name, tuple):
                scrubbed_names.append(name[0])
            else:
                scrubbed_names.append(name)
        self.classes = scrubbed_names

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index



class ImageNet1k(ImageNet):
    """Overwrite torchvision ImageNet to limit it to less than 1mio examples.

    [limit/per class, due to automl restrictions].
    """

    def __init__(self, root, split='train', download=False, limit=950, **kwargs):
        """As torchvision.datasets.ImageNet except for additional keyword 'limit'."""
        super().__init__(root, split, download, **kwargs)

        # Dictionary, mapping ImageNet1k ids to ImageNet ids:
        self.full_imagenet_id = dict()
        # Remove samples above limit.
        examples_per_class = torch.zeros(len(self.classes))
        new_samples = []
        new_idx = 0
        for full_idx, (path, target) in enumerate(self.samples):
            if examples_per_class[target] < limit:
                examples_per_class[target] += 1
                item = path, target
                new_samples.append(item)
                self.full_imagenet_id[new_idx] = full_idx
                new_idx += 1
            else:
                pass
        self.samples = new_samples
        print(f'Size of {self.split} dataset reduced to {len(self.samples)}.')




"""
    The following class is heavily based on code by Meng Lee, mnicnc404. Date: 2018/06/04
    via
    https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
"""


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Author: Meng Lee, mnicnc404
    Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    BASE_FOLDER = 'tiny-imagenet-200'

    def __init__(self, root, split='train', transform=None, target_transform=None):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(os.path.join(root, self.BASE_FOLDER))
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(self.root, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a triplet of image, label, index."""
        file_path, target = self.image_paths[index], self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == 'test':
            return img, None, index
        else:
            return img, target, index


    def get_target(self, index):
        """Return only the target and its id."""
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
