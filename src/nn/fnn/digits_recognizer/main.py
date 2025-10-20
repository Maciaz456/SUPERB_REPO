'''Main script to run digits recognition.'''
import sys
from pathlib import Path

import torch

sys.path.append(
    f'{Path(__file__).parent.parent.parent.parent}'
)
from my_logger.my_logger import MyLogger
from nn.digits_recognizer.dr_common import get_args
from nn.digits_recognizer.digits_recognizer import DigitsRecognizer


if __name__ == '__main__':
    args = get_args()

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    )

    logger = MyLogger(
        level=args.log_level,
        command_line=True,
        log_file=args.log_file
    )

    model = DigitsRecognizer(
        device,
        args.hidden_sizes,
        logger
    )

    if args.learn:
        model.train()

        model.prepare_mnist_dataset(
            args.dataset_folder,
            args.batch_size
        )

        model.learn(
            args.learning_rate,
            args.epochs,
            args.accuracy_threshold,
            args.pth_file
        )

    if args.evaluate:
        model.eval()
        if args.read_pth_file:
            state_dict = torch.load(
                args.pth_file
            )
            model.load_state_dict(
                state_dict
            )
        with torch.no_grad():
            model.recognize(
                args.image_paths
            )
