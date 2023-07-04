import math
import random
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

__all__ = [
    "AutoAugmentPolicy",
    "AutoAugment",
    "RandAugment",
    "TrivialAugmentWide",
    "EventMix",
    "event_drop",
    "Cutout",
]


class Cutout:
    """Randomly mask out one or more patches from an image.
    Altered from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
        img (Tensor): Tensor image of size (C, H, W).
        Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)

        mask = torch.ones((h, w)).type_as(img)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

            mask = mask.expand_as(img)
            img = img * mask

        return img


# EventDrop augmentation by randomly dropping events
def event_drop(events, resolution):
    raw_events = events
    option = np.random.randint(
        0, 4
    )  # 0: identity, 1: drop_by_time, 2: drop_by_area, 3: random_drop
    if option == 0:  # identity, do nothing
        return events
    elif option == 1:  # drop_by_time
        T = np.random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
        events = drop_by_time(events, T=T)
    elif option == 2:  # drop by area
        area_ratio = (
            np.random.randint(1, 20) / 20.0
        )  # np.random.uniform(0.05, 0.1, 0.15, 0.2, 0.25)
        events = drop_by_area(events, resolution=resolution, area_ratio=area_ratio)
    elif option == 3:  # random drop
        ratio = np.random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
        events = random_drop(events, ratio=ratio)

    if len(events) == 0:  # avoid dropping all the events
        events = raw_events

    return events


# drop the events within a given time interval
# param: events contains x, y, t, p information
def drop_by_time(events, T=0):
    # assert 0.1 <= T <= 0.5

    # time interval
    t_start = np.random.uniform(0, 1)
    if T == 0:  # randomly choose a value between [0.1, 0.9]
        T = np.random.randint(1, 10) / 10.0
    t_end = t_start + T

    timestamps = events[:, 0]
    max_t = max(timestamps)
    idx = (timestamps < (max_t * t_start)) | (
        timestamps > (max_t * t_end)
    )  # remaining events that are not within the given time interval

    return events[idx]


# drop the events within a a fixed area constrained by X, Y
# area ratio: the ratio between the area whose pixels are dropped and the overall pixel range
def drop_by_area(events, resolution=(128, 128), area_ratio=0):
    # assert 0.1 <= area_ratio <= 0.3

    # get the area whose events are to be dropped
    x0 = np.random.uniform(resolution[0])
    y0 = np.random.uniform(resolution[1])

    if area_ratio == 0:
        area_ratio = np.random.randint(1, 6) / 20.0

    x_out = resolution[0] * area_ratio
    y_out = resolution[1] * area_ratio

    x0 = int(max(0, x0 - x_out / 2.0))
    y0 = int(max(0, y0 - y_out / 2.0))

    x1 = min(resolution[0], x0 + x_out)
    y1 = min(resolution[1], y0 + y_out)

    xy = (x0, x1, y0, y1)  # rectangele to be dropped

    idx1 = (events[:, 2] < xy[0]) | (events[:, 2] > xy[1])
    idx2 = (events[:, 3] < xy[2]) | (events[:, 3] > xy[3])
    idx = idx1 & idx2

    return events[idx]


# randomly drop a proportion of events
def random_drop(events, ratio=0):
    # assert 0.1 <= ratio <= 0.5

    if ratio == 0:
        ratio = np.random.randint(1, 10) / 10.0

    N = events.shape[0]  # number of total events
    num_drop = int(N * ratio)  # number of events to be dropped
    idx = random.sample(list(np.arange(0, N)), N - num_drop)

    return events[idx]


# randomly shift events
def random_shift_events(events, max_shift=20, resolution=(128, 128), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
        events[:, 2] += x_shift
        events[:, 3] += y_shift
        valid_events = (
            (events[:, 2] >= 0)
            & (events[:, 2] < W)
            & (events[:, 3] >= 0)
            & (events[:, 3] < H)
        )
        events = events[valid_events]
    return events


# randomly flip events
def random_flip_events_along_x(events, resolution=(128, 128), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:, 2] = W - 1 - events[:, 2]
    return events


class Resize:
    def __init__(self, patch_size=64) -> None:
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(size=(patch_size, patch_size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, x):
        new_data = []
        for t in range(x.shape[0]):
            new_data.append(self.resize(torch.from_numpy(x[t, ...])))
        data = torch.stack(new_data, dim=0)
        return data


class EventMix:
    def __init__(self, sensor_size, T=8, num_classes=100, mode="distance"):
        h, w, _ = sensor_size
        self.sensor_size = (h, w)
        self.T = T
        self.num_classes = num_classes
        mask = self._gen_mask()
        self.mask = torch.from_numpy(mask.reshape((T, 1, h, w)))
        self.mode = mode

    def mix(self, frames, labels):
        # frames, should be N, T, C, H, W && N > 1
        # labels, should be N, 1          && N > 1
        frames_rolled = frames.roll(1, 0)

        # -----------------------------
        # Based on the number of events
        # -----------------------------
        self.mask = self.mask.to(frames.device)
        if self.mode == "events":
            sum_A = torch.mul(self.mask, frames).sum() / frames.sum()
            sum_B = torch.mul(1 - self.mask, frames_rolled).sum() / frames_rolled.sum()
            alpha = sum_A / (sum_A + sum_B)
        elif self.mode == "distance":
            # ------------------------------
            # Based on the relative distance
            # ------------------------------
            # print(frames.shape, self.mask.shape)
            x_mean = F.adaptive_avg_pool2d(
                frames.mul_(self.mask) + frames_rolled.mul_(1 - self.mask), 1
            )
            frames_pooled = F.adaptive_avg_pool2d(frames, 1)
            frames_rolled_pooled = F.adaptive_avg_pool2d(frames_rolled, 1)

            distance = nn.MSELoss(reduction="sum")
            sum_A = distance(frames_pooled, x_mean).item() ** 2
            sum_B = distance(frames_rolled_pooled, x_mean).item() ** 2
            alpha = sum_B / (sum_A + sum_B)
        # print(alpha)

        lambda_param = float(torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0])

        # labels mix
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
            dtype=frames.dtype
        )
        labels_rolled = labels_onehot.roll(1, 0)
        labels_rolled.mul_(1.0 - lambda_param)
        labels_onehot.mul_(lambda_param).add_(labels_rolled)

        # frames mix
        frames_rolled.mul_(1.0 - lambda_param)
        frames.mul_(lambda_param).add_(frames_rolled)

        return frames, labels_onehot

    def _gen_mask(self):
        mean = np.random.randn(self.T)
        cov = np.random.randn(self.T, 1)
        cov = 0.5 * cov * cov.T
        mask = np.random.multivariate_normal(mean, cov, self.sensor_size).reshape(
            -1,
        )
        lam = int(np.random.beta(1, 1) * mask.size)
        threshold = mask[np.argpartition(mask, kth=-lam)[-lam]]
        for i, value in enumerate(mask):
            if value < threshold:
                mask[i] = 1
            else:
                mask[i] = 0
        return mask


class SNNAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(
                magnitudes[
                    torch.randint(len(magnitudes), (1,), dtype=torch.long)
                ].item()
            )
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        return _apply_op(
            img, op_name, magnitude, interpolation=self.interpolation, fill=fill
        )

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)

    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """

    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


# FIXME: Eliminate copy-pasted code for fill standardization and _augmentation_space() by moving stuff on a base class
class AutoAugment(torch.nn.Module):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[
        Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]
    ]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError("The provided policy {} is not recognized.".format(policy))

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation
        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                op_meta = self._augmentation_space(10, F.get_image_size(img))
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[magnitude_id].item())
                    if magnitude_id is not None
                    else 0.0
                )
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = _apply_op(
                    img, op_name, magnitude, interpolation=self.interpolation, fill=fill
                )

        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(policy={}, fill={})".format(
            self.policy, self.fill
        )


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
                True,
            ),
            "TranslateY": (
                torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
                True,
            ),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(
                self.num_magnitude_bins, F.get_image_size(img)
            )
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(
                img, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )

        return img

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_ops={num_ops}"
        s += ", magnitude={magnitude}"
        s += ", num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)


class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(
                magnitudes[
                    torch.randint(len(magnitudes), (1,), dtype=torch.long)
                ].item()
            )
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(
            img, op_name, magnitude, interpolation=self.interpolation, fill=fill
        )

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)
