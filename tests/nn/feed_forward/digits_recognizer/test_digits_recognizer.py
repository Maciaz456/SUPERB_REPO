from unittest.mock import Mock, call, patch

import pytest
import torch

from nn.ff.digits.digits import Digits, N_PIXELS, N_CLASSES
from nn.ff.digits.d_common import DigitsException


@pytest.fixture
def dummy_device():
    return torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )


@pytest.fixture(
    params=[
        [500, 200]
    ]
)
def dummy_digits(
    dummy_device,
    request,
    dummy_my_logger
):
    return Digits(
        dummy_device,
        request.param,
        dummy_my_logger
    )


@pytest.fixture
def dummy_tensor():
    def subfunction(
        seed
    ):
        torch.manual_seed(
            seed
        )
        dummy_tensor = torch.rand(
            (28, 28),
            dtype=torch.float32
        )

        return dummy_tensor

    return subfunction


@patch(
    'nn.ff.digits.digits.torch.nn.ReLU', spec_set=torch.nn.ReLU
)
@patch(
    'nn.ff.digits.digits.torch.nn.Linear', spec_set=torch.nn.Linear
)
def test_layers_setup(
    linear_mock,
    relu_mock,
    dummy_device,
    dummy_my_logger
):
    hidden_sizes = [
        500,
        100
    ]

    Digits(
        dummy_device,
        hidden_sizes,
        dummy_my_logger
    )

    assert linear_mock.call_args_list == [
        call(
            N_PIXELS, 500
        ),
        call(
            500, 100
        ),
        call(
            100, N_CLASSES
        )
    ]

    assert len(
        relu_mock.call_args_list
    ) == len(
        hidden_sizes
    )


@patch(
    'nn.ff.digits.digits.torch.Tensor.to',
    return_value='to_return'
)
def test_forward(
    to_mock,
    dummy_digits,
    dummy_tensor
):
    dummy_digits._Digits__layers.forward = Mock(
        return_value=dummy_tensor(1)
    )

    return_value = dummy_digits(
        dummy_tensor(1)
    )

    to_mock.assert_called_once_with(
        dummy_digits._Digits__device
    )

    dummy_digits._Digits__layers.forward.assert_called_once_with(
        to_mock.return_value
    )
    assert (
        return_value == dummy_digits._Digits__layers.forward.return_value
    ).all()


@patch(
    'nn.ff.digits.digits.datasets.MNIST'
)
@patch(
    'nn.ff.digits.digits.torch.utils.data.DataLoader'
)
def test_mnist_dataset(
    data_loader_mock,
    mnist_mock,
    dummy_digits,
    tmp_path
):
    batch_size = 100

    dummy_digits.prepare_mnist_dataset(
        tmp_path,
        batch_size=batch_size
    )

    assert mnist_mock.call_args_list == [
        call(
            root=tmp_path,
            train=True,
            transform=dummy_digits._transform,
            download=True
        ),
        call(
            root=tmp_path,
            train=False,
            transform=dummy_digits._transform
        )
    ]

    assert data_loader_mock.call_args_list == [
        call(
            dataset=dummy_digits._train_dataset,
            batch_size=batch_size,
            shuffle=True
        ),
        call(
            dataset=dummy_digits._test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    ]


def test_create_image_tensor(
    dummy_digits,
    data_folder
):
    img_path = data_folder.joinpath(
        'correct.png'
    )
    image_tensor = dummy_digits.create_image_tensor(
        img_path
    )
    correct_tensor_file = data_folder.joinpath(
        'correct_tensor.pt'
    )
    correct_tensor = torch.load(
        correct_tensor_file
    )
    assert torch.equal(
        image_tensor,
        correct_tensor
    )

    img_path = data_folder.joinpath(
        'incorrect.png'
    )
    with pytest.raises(
        DigitsException
    ):
        dummy_digits.create_image_tensor(
            img_path
        )
