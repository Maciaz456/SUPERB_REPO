from unittest.mock import Mock, call, patch

import pytest
import torch

from nn.ff.digits.digits import Digits, N_PIXELS, N_CLASSES
from nn.ff.digits.d_common import DigitsException


@pytest.fixture
def device():
    return torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )


@pytest.fixture(
    params=[
        [500, 200]
    ]
)
def digits(
    device,
    request,
    dummy_my_logger
):
    return Digits(
        device,
        request.param,
        dummy_my_logger
    )


@pytest.fixture
def dummy_tensor():
    return torch.ones(
        (28, 28),
        dtype=torch.float32
    )


@pytest.mark.parametrize(
    'hidden_sizes',
    argvalues=[
        [500],
        [200, 100]
    ],
    ids=[
        'one_hidden_layer',
        'two_hidden_layers'
    ]
)
def test_layers_setup(
    device,
    hidden_sizes,
    dummy_my_logger
):
    digits = Digits(
        device,
        hidden_sizes,
        dummy_my_logger
    )

    layers = digits._Digits__layers
    n_layers = len(
        layers
    )
    for i in range(
        n_layers
    ):

        if i == 0:
            assert layers[i].in_features == N_PIXELS
        elif i == len(layers) - 1:
            assert layers[i].out_features == N_CLASSES

        if i < n_layers - 1:
            assert layers[i].out_features == layers[i+1].in_features

    assert n_layers == len(hidden_sizes) + 1


def test_forward(
    digits,
    dummy_tensor
):
    for i, layer in enumerate(
        digits._Digits__layers
    ):
        mocked_layer = Mock(
            spec_set=layer,
            return_value=dummy_tensor
        )
        digits._Digits__layers[i] = mocked_layer

    digits._relu = Mock(
        spec_set=digits._relu
    )

    digits.forward(
        dummy_tensor
    )
    for layer in digits._Digits__layers:
        layer.assert_called_once()
    assert digits._relu.call_count == len(digits._Digits__layers) - 1


@patch(
    'nn.ff.digits.digits.datasets.MNIST'
)
@patch(
    'nn.ff.digits.digits.torch.utils.data.DataLoader'
)
def test_mnist_dataset(
    data_loader_mock,
    mnist_mock,
    digits,
    tmp_path
):
    batch_size = 100

    digits.prepare_mnist_dataset(
        tmp_path,
        batch_size=batch_size
    )

    assert mnist_mock.call_args_list == [
        call(
            root=tmp_path,
            train=True,
            transform=digits._transform,
            download=True
        ),
        call(
            root=tmp_path,
            train=False,
            transform=digits._transform
        )
    ]

    assert data_loader_mock.call_args_list == [
        call(
            dataset=digits._train_dataset,
            batch_size=batch_size,
            shuffle=True
        ),
        call(
            dataset=digits._test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    ]


def test_create_image_tensor(
    digits,
    data_folder
):
    img_path = data_folder.joinpath(
        'correct.png'
    )
    image_tensor = digits.create_image_tensor(
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
        digits.create_image_tensor(
            img_path
        )
