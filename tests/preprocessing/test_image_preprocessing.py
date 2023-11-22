import numpy as np
import pytest
import torch
from PIL import Image

from photoholmes.preprocessing.image import RGBtoGray, ToNumpy, ToTensor


# ============================= ToTensor Tests =========================================
@pytest.fixture
def to_tensor():
    return ToTensor()


def test_to_tensor_3_channels(to_tensor: ToTensor):
    np_image = np.random.rand(100, 100, 3).astype(np.float32)
    dct_coeffs = np.random.rand(100, 100, 2).astype(np.float32)

    result = to_tensor(np_image, dct_coefficients=dct_coeffs)

    assert isinstance(result, dict)
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["dct_coefficients"], torch.Tensor)
    assert result["image"].shape == (3, 100, 100)
    assert result["dct_coefficients"].shape == (100, 100, 2)

    np.testing.assert_allclose(result["image"].numpy(), np_image.transpose((2, 0, 1)))
    np.testing.assert_allclose(result["dct_coefficients"].numpy(), dct_coeffs)


def test_to_tensor_1_channel(to_tensor: ToTensor):
    np_image = np.random.rand(100, 100).astype(np.float32)

    result = to_tensor(np_image)

    assert isinstance(result, dict)
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape == (100, 100)

    np.testing.assert_allclose(result["image"].numpy(), np_image)


# =============================== Normalize Tests ======================================
@pytest.fixture
def to_numpy():
    return ToNumpy()


def test_to_numpy_tensor(to_numpy: ToNumpy):
    # Create a torch tensor
    torch_image = torch.rand((3, 100, 100))

    # Apply the ToNumpy transform
    result = to_numpy(torch_image)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["image"], np.ndarray)
    assert result["image"].shape == (3, 100, 100)

    # Check that the values in the array are the same as in the tensor
    np.testing.assert_allclose(result["image"], torch_image.numpy())


def test_to_numpy_pil(to_numpy: ToNumpy):
    # Create a PIL Image
    pil_image = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))

    # Apply the ToNumpy transform
    result = to_numpy(pil_image)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["image"], np.ndarray)
    assert result["image"].shape == (100, 100, 3)

    # Check that the values in the array are the same as in the PIL Image
    np.testing.assert_allclose(result["image"], np.array(pil_image))


def test_to_numpy_kwargs_tensor(to_numpy: ToNumpy):
    # Create a torch tensor
    extra = torch.zeros((6,))

    # Apply the ToNumpy transform
    result = to_numpy(extra=extra)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["extra"], np.ndarray), "Extra should come out as an array"
    assert result["extra"].shape == (6,)
    assert np.allclose(result["extra"], extra.numpy())


def test_to_numpy_kwargs_numpy(to_numpy: ToNumpy):
    # Create a numpy array
    extra = np.zeros((6,))

    # Apply the ToNumpy transform
    result = to_numpy(extra=extra)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["extra"], np.ndarray), "Extra should come out as an array"
    assert result["extra"].shape == (6,)
    assert np.allclose(result["extra"], extra)


def test_to_numpy_kwargs_other(to_numpy: ToNumpy):
    # Create a list
    extra = [0, 1, 2, 3, 4, 5]

    # Apply the ToNumpy transform
    result = to_numpy(extra=extra)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["extra"], np.ndarray), "Extra should come out as an array"
    assert result["extra"].shape == (6,)
    assert np.allclose(result["extra"], np.array(extra))


# =============================== RGBtoGray Tests ======================================
@pytest.fixture
def rgb_to_gray():
    return RGBtoGray()


def test_rgb_to_gray_tensor(rgb_to_gray: RGBtoGray):
    # Create a torch tensor
    torch_image = torch.rand((3, 100, 100))
    extra = "passthrough"

    # Apply the RGBtoGray transform
    result = rgb_to_gray(torch_image, extra=extra)

    # Check that the output is a dictionary with a torch tensor
    assert isinstance(result, dict)
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["extra"], str), "Extra should be passed through unchanged"
    assert result["image"].shape == (100, 100)


def test_rgb_to_gray_numpy(rgb_to_gray: RGBtoGray):
    # Create a numpy array
    np_image = np.random.rand(100, 100, 3).astype(np.float32)
    extra = "passthrough"

    # Apply the RGBtoGray transform
    result = rgb_to_gray(np_image, extra=extra)

    # Check that the output is a dictionary with a numpy array
    assert isinstance(result, dict)
    assert isinstance(result["image"], np.ndarray)
    assert isinstance(result["extra"], str), "Extra should be passed through unchanged"
    assert result["image"].shape == (100, 100)


# TODO add tests for Normalize when EXIF is merged
