'''Contain DigitsRecognizer.'''
import os
from annotated_types import Ge, Le
from pathlib import Path
from typing import Annotated

import torch
from PIL import Image
from torchvision import datasets, transforms

from nn.feed_forward.digits_recognizer.dr_common import DigitsRecognizerException, custom_validate_call
from my_logger.my_logger import MyLogger


N_PIXELS = 784
N_CLASSES = 10


class DigitsRecognizer(
    torch.nn.Module
):
    '''Recognize digits on images with the size of 28x28 pixels.'''
    @custom_validate_call
    def __init__(
        self,
        device: torch.device,
        hidden_sizes: list[Annotated[int, Ge(1)]],
        logger: MyLogger
    ):
        '''
        :param device:        Device (CPU/CUDA) where computations will be carried out.
        :param hidden_sizes:  Sizes of the hidden layers.
        :param logger:        MyLogger instance to log messages.
        '''
        super().__init__()

        self.__layers = torch.nn.ModuleList()
        in_features = N_PIXELS
        for layer_size in hidden_sizes:
            self.__layers.append(
                torch.nn.Linear(
                    in_features,
                    layer_size
                )
            )
            in_features = layer_size

        self.__layers.append(
            torch.nn.Linear(
                in_features,
                N_CLASSES
            )
        )

        self._relu = torch.nn.ReLU()

        self._loss_fn = torch.nn.CrossEntropyLoss()

        self._transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        self.__device = device
        self.to(
            device
        )

        self._logger = logger

    @custom_validate_call
    def forward(
        self,
        image_tensors: torch.Tensor
    ) -> torch.Tensor:
        '''
        Forward the input through the model.

        :params image_tensors:  Input tensor of images converted to tensors.

        :return:                Tensor of the output logits.
        '''
        out = image_tensors.to(
            self.__device
        )
        for i, layer in enumerate(
            self.__layers
        ):
            out = layer(
                out
            )
            if i < len(
                self.__layers
            ) - 1:
                out = self._relu(
                    out
                )

        return out

    @custom_validate_call
    def prepare_mnist_dataset(
        self,
        root_folder: Path,
        batch_size: Annotated[int, Ge(1)]
    ) -> None:
        '''
        Handle preparing the MNIST dataset. Download it and create datasets with loaders.

        :param root_folder:  Root folder where the MNIST dataset will be downloaded.
        :param batch_size:   Batch size for the train and test loaders.
        '''
        self._logger.info(
            'Preparing the MNIST dataset.'
        )

        if root_folder.exists() and not root_folder.is_dir():
            raise DigitsRecognizerException(
                (
                    'Root folder for the MNIST dataset cannot be created, '
                    f'because a file with the same path exists: {root_folder}'
                )
            )
        self._train_dataset = datasets.MNIST(
            root=root_folder,
            train=True,
            transform=self._transform,
            download=True
        )
        self._test_dataset = datasets.MNIST(
            root=root_folder,
            train=False,
            transform=self._transform
        )

        self._train_loader = torch.utils.data.DataLoader(
            dataset=self._train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self._test_loader = torch.utils.data.DataLoader(
            dataset=self._test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self._logger.debug(
            'Preparing the MNIST dataset completed.'
        )

    @custom_validate_call
    def learn(
        self,
        lr: Annotated[float, Ge(0)],
        epochs: Annotated[int, Ge(1)],
        accuracy_threshold: Annotated[float, Ge(0), Le(100)] | None = None,
        pth_file: Path | None = None
    ) -> None:
        '''
        Run model learning.

        :param lr:                  Learning rate for the optimizer.
        :param epochs:              Number of epochs in the training loop.
        :param accuracy_threshold:  Acceptable accuracy threshold to save the model parameters.
        :param pth_file:           .PTH file to save the model parameters.
        '''
        self._logger.info(
            'Learning the model.'
        )

        self.__train(
            lr,
            epochs,
        )

        self.__test(
            accuracy_threshold,
            pth_file
        )

        self._logger.debug(
            'Learning the model completed.'
        )

    @custom_validate_call
    def __train(
        self,
        lr: Annotated[float, Ge(0)],
        epochs: Annotated[int, Ge(1)]
    ) -> None:
        '''
        Train the model.

        :param lr:            Learning rate for the optimizer.
        :param epochs:        Number of epochs in the training loop.
        '''
        self._logger.info(
            'Training the model.'
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr
        )

        for epoch in range(
            epochs
        ):
            summed_loss = 0.0
            for samples, labels in self._train_loader:
                optimizer.zero_grad()

                samples = samples.reshape(
                    -1, N_PIXELS
                )

                prediction = self(
                    samples
                )

                labels = labels.to(
                    self.__device
                )
                loss = self._loss_fn(
                    prediction,
                    labels
                )

                loss.backward()
                optimizer.step()

                summed_loss += loss.item()

            avg_loss = summed_loss / len(self._train_loader)
            self._logger.debug(
                f'Epoch: {epoch}, average batch loss: {avg_loss:.4f}'
            )

        self._logger.debug(
            'Training the model completed.'
        )

    @custom_validate_call
    def __test(
        self,
        accuracy_threshold: Annotated[float, Ge(0), Le(100)] | None = None,
        pth_file: Path | None = None
    ) -> None:
        '''
        Test the model.

        :param accuracy_threshold:  Acceptable accuracy threshold to save the model parameters.
                                    If not passed, the model parameters will not be saved.
        :param pth_file:            .PTH file to save the model parameters. It will be removed if it already exists.
        '''
        self._logger.info(
            'Testing the model.'
        )

        total = 0
        successful = 0

        with torch.no_grad():
            for samples, labels in self._test_loader:
                samples = samples.reshape(
                    -1, N_PIXELS
                )
                total += samples.shape[0]

                prediction = self(
                    samples
                )
                prediction = prediction.argmax(
                    1
                )

                labels = labels.to(
                    self.__device
                )
                successful += (prediction == labels).sum().item()

        test_accuracy = successful / total
        self._logger.info(
            f'Test accuracy: {(successful / total):.1%}'
        )
        if accuracy_threshold is not None and test_accuracy*100 >= accuracy_threshold:
            if pth_file and pth_file.exists():
                os.unlink(
                    pth_file
                )
            elif not pth_file:
                raise DigitsRecognizerException(
                    f'Specified .PTH file path to save the model parameters is incorrect: {pth_file}'
                )
            torch.save(
                self.state_dict(),
                pth_file
            )

        self._logger.debug(
            'Testing the model completed.'
        )

    @custom_validate_call
    def recognize(
        self,
        img_paths: list[Path]
    ) -> dict[Path, Annotated[int, Ge(0), Le(9)]]:
        '''
        Recognize digits on the images.

        :param img_paths:  List of image paths.

        :return:           List of the recognized digits.
        '''
        self._logger.info(
            'Recognizing digits.'
        )

        result = dict()

        for img_path in img_paths:
            self._logger.info(
                f'Image: {img_path}'
            )
            img_tensor = self.create_image_tensor(
                img_path
            )
            digit = self(
                img_tensor
            )
            digit = digit.argmax().item()

            self._logger.info(
                f'Digit: {digit}'
            )
            result[img_path] = digit

        self._logger.info(
            'Recognizing digits completed.'
        )

        return result

    @custom_validate_call
    def create_image_tensor(
        self,
        img_path: Path
    ) -> torch.Tensor:
        '''
        Create tensor from the image.

        :param img_path:  Path to the image.

        :return:          Image tensor.
        '''
        self._logger.debug(
            'Creating a tensor from an image.'
        )

        if not img_path.exists():
            raise FileNotFoundError(
                f'Following image to recognize does not exist: {img_path}!'
            )

        img = Image.open(
            img_path
        )
        img = img.convert(
            'L'
        )
        img = self._transform(
            img
        )
        if not img.shape == (
            1, 28, 28
        ):
            raise DigitsRecognizerException(
                f'{img_path} has incorrect shape: {img.shape}!'
            )

        img = img.to(
            self.__device
        )
        img = img.reshape(
            -1,
            N_PIXELS
        )

        self._logger.debug(
            'Creating a tensor from the image completed.'
        )

        return img
