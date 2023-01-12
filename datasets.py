from PIL import Image
import natsort
import os
from itertools import groupby

from torchvision import datasets, transforms
from torch.utils.data import Dataset

preprocessing = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1)])
#                   transforms.Normalize([0.5094], [0.2314])])


class AugmentedBalancedDataset(Dataset):
    def __init__(self, augment, rotation=0, translation_h=0.00,
                 translation_v=0.00, fill=0, scale=0.0,
                 hfp=0, foldername="xray-data/xray-data/train"):
        """A custom dataset that expands the size of train dataset
            and applies desired transforms to it. Includes balancing
            of size of data in each class using data augmentation.

        Args:
            augment (int): factor by which you would like to expand the dataset
            rotation (int, optional): Range of angle for rotation.
                                        Defaults to 0.
            translation_h (float, optional): maximum absolute fraction for
                                                horizontal translation.
                                                Defaults to 0.00.
            translation_v (float, optional): maximum absolute fraction for
                                                vertical translation.
                                                Defaults to 0.00.
            scale (float, optional): Value to add and subtract from 1 to
                                        define scaling factor interval.
                                        Defaults to 0.0.
            hfp (int, optional): Probability to flip horizontally.
                                    Defaults to 0.
            gamma (int, optional): Gamma value for gamma filter. Defaults to 1.
            gaussian (float, optional): Standard Deviation to apply to gaussian
                                        blur. Defaults to 0.001.
            foldername (str, optional): Path of file storing data to load
                                        using ImageFolder. Defaults to
                                        "xray-data/xray-data/train".
        """
        self.dataset = datasets.ImageFolder(foldername,
                                            transform=preprocessing)
        self.labels = [i[1] for i in self.dataset]
        self.truelen = len(self.labels)  # Length of our original dataset

        # By what factor are we augmenting the data?
        self.augment = augment

        # The original frequencies of our different classes
        self.freqs = [len(list(group)) for key,
                      group in groupby(self.labels)]
        # How many instances of our most frequent class?
        self.maxlen = max(self.freqs)
        # The length that the (padded) augmented set should have
        self.nmlen = self.maxlen*4

        # save all the kwargs
        self.hfp = hfp
        self.rotation = rotation
        self.translation = (translation_h, translation_v)
        self.fill = fill
        self.scale = (1-scale, 1+scale)

        # Defining our random transform
        self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=self.hfp),
                            transforms.RandomAffine(self.rotation,
                                                    translate=self.translation,
                                                    fill=self.fill,
                                                    scale=self.scale)])

    def __len__(self):
        return self.nmlen*self.augment

    def __getitem__(self, idx):
        returnTr = False
        # If our index is bigger than our 36k padded dataset, then we are
        # going to return a transformed image anyways
        if idx > self.nmlen:
            returnTr = True

        idxr = idx % self.nmlen  # reduced idx within our 36k dataset
        ctr = idxr//self.maxlen  # Class to return

        # Once we know the class to return, we check whether we need any
        # padding: if we need padding, we return a transformed image
        if idxr % self.maxlen >= self.freqs[ctr]:
            returnTr = True

        # We have just determined whether we need to return a transformed or
        # non transformed image, so here we are just selecting the
        # image to return (transformed or not)
        newidx = (idxr % self.maxlen) % self.freqs[ctr]+sum(self.freqs[:ctr])
        # If we have to return a transformed image,
        # we return a transformed image, if not we don't.
        # The index of the image is cyclically picked from our original non
        # padded data for the class we have to return
        if returnTr:
            return self.transform(self.dataset[int(newidx)][0]), ctr
        else:
            return self.dataset[int(newidx)][0], ctr


class AugmentedDataset(Dataset):
    def __init__(self, augment, rotation=0, translation_h=0.00,
                 translation_v=0.00, scale=0.0, hfp=0,
                 gamma=1, gaussian=0.001,
                 foldername="xray-data/xray-data/train"):
        """A custom dataset that expands the size of train dataset
        and applies desired transforms to it.

        Args:
            augment (int): factor by which you would like to expand the dataset
            rotation (int, optional): Range of angle for rotation.
                                      Defaults to 0.
            translation_h (float, optional): maximum absolute fraction for
                                             horizontal translation.
                                             Defaults to 0.00.
            translation_v (float, optional): maximum absolute fraction for
                                             vertical translation.
                                             Defaults to 0.00.
            scale (float, optional): Value to add and subtract from 1 to
                                     define scaling factor interval.
                                     Defaults to 0.0.
            hfp (int, optional): Probability to flip horizontally.
                                 Defaults to 0.
            gamma (int, optional): Gamma value for gamma filter. Defaults to 1.
            gaussian (float, optional): Standard Deviation to apply to gaussian
                                        blur. Defaults to 0.001.
            foldername (str, optional): Path of file storing data to load
                                        using ImageFolder. Defaults to
                                        "xray-data/xray-data/train".
        """
        self.dataset = datasets.ImageFolder(foldername,
                                            transform=preprocessing)
        self.targets = self.dataset.targets*augment
        self.samples = self.dataset.samples*augment
        # Length of our original dataset
        self.truelen = len(self.dataset.targets)

        # By what factor are we augmenting the data?
        self.augment = augment

        # save all the kwargs
        self.hfp = hfp
        self.rotation = rotation
        self.translation = (translation_h, translation_v)
        self.scale = (1 - scale, 1 + scale)
        self.gamma = gamma  # gamma value for the gamma filter, default = 1
        self.gaussian = gaussian  # sigma value for gaussian

        # define gamma filter transform
        gamma_filter = transforms.Lambda(lambda x:
                                         transforms.functional.adjust_gamma
                                         (x, self.gamma))

        # define gaussian blur
        # gaussian_blur = transforms.GaussianBlur(kernel_size=101,
        #                                        sigma=(0.001, self.gaussian))
        # range of std is 0 to given max value

        # Defining our random transform
        self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=self.hfp),
                            transforms.RandomAffine(self.rotation,
                                                    translate=self.translation,
                                                    scale=self.scale),
                            gamma_filter
                          ])

    def __len__(self):
        return self.truelen*self.augment

    def __getitem__(self, idx):
        """
        If the index is outside of our true length, return
        a transformed image. Else return the "original"
        """
        if idx >= self.truelen:
            return self.transform(self.dataset[idx % self.truelen][0]), \
                   self.dataset[idx % self.truelen][1]
        else:
            return self.dataset[idx]


class TestDataSet(Dataset):
    """ Generate a TestDataSet. Inspired by:
    https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999
    """
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
