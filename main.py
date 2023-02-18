import argparse
import os
import torch
from models.discriminator import Discriminator
from dataset.datasetgetter import get_dataset
from train.train import train_style_discriminator


def main():
    parser = argparse.ArgumentParser(description='PyTorch GAN Training')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../refined-fonts-images',
        help='Dataset directory. Please refer Dataset in README.md')
    parser.add_argument(
        '--epochs',
        default=250,
        type=int,
        help='Total number of epochs to run. Not actual epoch.')
    parser.add_argument(
        '--iters',
        default=1000,
        type=int,
        help='Total number of iterations per epoch')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--gpu', action='store_true',
                        help='Using GPU or not')
    parser.add_argument('--save_dir', default='checkpoints',
                        help='The size of data pair for discriminator')
    parser.add_argument('--input_ch', default=1,
                        help='The number of channels of input image')
    parser.add_argument('--pair_size', default=2,
                        help='The size of data pair for discriminator')
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help='Learning Rate')
    parser.add_argument(
        '--workers',
        default=8,
        type=int,
        help='Number of workers for data loader')
    parser.add_argument(
        '--check_point_step',
        default=1,
        type=int,
        help='Check point epoch step')
    parser.add_argument(
        '--log_step',
        default=1,
        type=int,
        help='Log iter step')

    args = parser.parse_args()
    train(args)


def train(args):
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    networks, opts = build_model(args)

    dataset = get_dataset(args)

    data_loader = get_loader(
        args,
        dataset,
        shuffle=True)

    for epoch in range(0, args.epochs):
        print("START EPOCH[{}]".format(epoch + 1))
        train_style_discriminator(data_loader, networks, opts, epoch, args)

        if (epoch + 1) % (args.check_point_step) == 0:
            save_model(args, epoch, networks, opts)

        print("\nFINISH EPOCH[{}]".format(epoch + 1))


def build_model(args):

    networks = {}
    opts = {}
    networks['D'] = Discriminator(
        image_size=args.img_size,
        input_ch=args.input_ch,
    )

    if args.device == 'cuda':
        # torch.cuda.set_device(args.device)
        for name, net in networks.items():
            # networks[name] = net.cuda(args.gpu)
            # networks[name] = net.cuda(args.device)
            networks[name] = net.to(args.device)

    opts['D'] = torch.optim.RMSprop(
        networks['D'].parameters(), args.lr, weight_decay=0.0001)
    return networks, opts


def save_checkpoint(state, save_dir, epoch=0):
    check_point_file = os.path.join(save_dir, f'model_{epoch}.ckpt')
    torch.save(state, check_point_file)


def save_model(args, epoch, networks, opts):
    with torch.no_grad():
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        for name, net in networks.items():
            save_dict[name + '_state_dict'] = net.state_dict()
            save_dict[name.lower() +
                      '_optimizer'] = opts[name].state_dict()
        print("SAVE CHECKPOINT[{}] DONE".format(epoch + 1))
        save_checkpoint(save_dict, args.save_dir, epoch + 1)


def get_loader(
        args,
        dataset,
        shuffle=True):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False)

    return data_loader
