"""
This code is based on the training code found at
https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
"""

import sys, os
import gc
import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch

import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from external.invertible_rim.irim.rim import RIM, ConvRNN
from external.invertible_rim.irim import IRIM, InvertibleUnet, ResidualBlockPixelshuffle

from external.fastMRI.common import evaluate as numpy_eval
from external.fastMRI.common.args import Args
from external.fastMRI.data import transforms

from training_utils.models import IRIMfastMRI, RIMfastMRI
from training_utils.helpers import mse_gradient, estimate_to_image, image_loss, real_to_complex, complex_to_real

from utils import get_cartesian_mask, get_kspace, kspace_to_image
from dataset import get_dataset, get_batches

torch.cuda.current_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def train_epoch(args, epoch, model, data_ds, optimizer, writer=None):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    mask_c = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)
    print('R = ', str(256 // args.n_keep))
    mask_c = torch.from_numpy(mask_c[None, None, :, :].astype(np.float32))
    n_steps = args.n_iters
    global_step = epoch * n_steps

    from_space = lambda x: kspace_to_image(x, (2, 3)).real
    memory_allocated = []
    i = 0
    for data in get_batches(args, data_ds):
        # y shape [BS, 1, 256, 256, 2]
        # mask shape [BS, 1, 1, 256, 1]
        # target shape [BS, 256, 256]
        target = data['image']
        target = torch.from_numpy(target.reshape(-1, 1, args.resolution, args.resolution).astype(np.float32))
        y = from_space(get_kspace(target, axes=(2, 3)) * mask_c)[:, :, :, :, None]
        y = y.repeat(1, 1, 1, 1, 2)
        # y = torch.cat((y, torch.zeros_like(y)), dim=-1)
        mask = mask_c[:, :, 1].clone() # mask shape [1, 1, 256]
        mask = mask[None, :, :, :, None].repeat(args.batch_size, 1, 1, 1, 1)
        target = target.reshape((-1, args.resolution, args.resolution))
        
        y = y.to(args.device)
        mask = mask.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        model.zero_grad()
        estimate = model.forward(y=y, mask=mask, metadata=None)
        # print('target: ', target.shape, 'estimate: ', estimate.shape)
        if isinstance(estimate, list):
            loss = [image_loss(e, target, args) for e in estimate]
            loss = sum(loss) / len(loss)
        else:
            loss = image_loss(estimate, target, args)

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if i > 0 else loss.item()           

        if args.device == 'cuda':
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.empty_cache()
        gc.collect()
        
        if dist.get_rank() == 0 and writer is not None:
            writer.add_scalar('Loss', loss.item(), global_step + i)
            if i % args.report_interval == 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{i:4d}/{n_steps:4d}] '
                    f'Loss = {loss.detach().item():.4g} Avg Loss = {avg_loss:.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s '
                    f'Memory allocated (MB) = {np.min(memory_allocated):.2f}'
                )
                memory_allocated = []
            start_iter = time.perf_counter()
            i+=1
            if i > n_steps:
                break
    
    optimizer.zero_grad()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_ds, writer):
    model.eval()
    mse_losses = []
    psnr_losses = []
    nmse_losses = []
    ssim_losses = []
    memory_allocated = []

    mask_c = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)
    mask_c = torch.from_numpy(mask_c[None, None, :, :].astype(np.float32))
    from_space = lambda x: kspace_to_image(x, (2, 3)).real

    start = time.perf_counter()
    n_steps = 25
    with torch.no_grad():
        i = 0
        for data in get_batches(args, data_ds):
            # y, mask, target, metadata = data[:4]
            target = data['image']
            target = torch.from_numpy(target.reshape(-1, 1, args.resolution, args.resolution).astype(np.float32))
            y = from_space(get_kspace(target, axes=(2, 3)) * mask_c)[:, :, :, :, None]
            y = y.repeat(1, 1, 1, 1, 2)
            # y = torch.cat((y, torch.zeros_like(y)), dim=-1)
            mask = mask_c[:, :, 1].clone() # mask shape [1, 1, 256]
            mask = mask[None, :, :, :, None].repeat(args.batch_size, 1, 1, 1, 1)
            target = target.reshape((-1, args.resolution, args.resolution))

            y = y.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            if args.n_slices > 1:
                output = model.forward(y=y, mask=mask, metadata=None)
                output = estimate_to_image(output, args.resolution)
                output_np = output.to('cpu').transpose(0, -4).squeeze(-4)
                del output
            else:
                # y = y.transpose(0, -4).squeeze(-4)
                # mask = mask.squeeze(-4).repeat(y.size(0), 1, 1, 1, 1)
                # metadata = metadata.repeat(y.size(0), 1)
                # print(y.shape, mask.shape)
                output_np = []
                # for k, l in zip(range(0, y.size(0), args.batch_size),
                #                 range(args.batch_size, y.size(0) + args.batch_size, args.batch_size)):
                # output = model.forward(y=y[k:l], mask=mask[k:l], metadata=None)
                output = model.forward(y=y, mask=mask)
                output = estimate_to_image(output, args.resolution)
                output_np.append(output.to('cpu'))
                output_np = torch.cat(output_np, 0)

            output_np = output_np.reshape(-1, output_np.size(-2), output_np.size(-1))
            target = target.reshape_as(output_np)

            output_np = output_np.to('cpu').numpy()
            target_np = target.to('cpu').numpy()
            mse_losses.append(numpy_eval.mse(target_np, output_np))
            psnr_losses.append(numpy_eval.psnr(target_np, output_np))
            nmse_losses.append(numpy_eval.nmse(target_np, output_np))
            ssim_losses.append(numpy_eval.ssim(target_np, output_np))

            if args.device == 'cuda':
                memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
                torch.cuda.reset_max_memory_allocated()

            del data, y, mask, target
            torch.cuda.empty_cache()
            i+=1
            if i > n_steps:
                break

        writer.add_scalar('Val_MSE', np.mean(mse_losses), epoch)
        writer.add_scalar('Val_PSNR', np.mean(psnr_losses), epoch)
        writer.add_scalar('Val_NMSE', np.mean(nmse_losses), epoch)
        writer.add_scalar('Val_SSIM', np.mean(ssim_losses), epoch)
        writer.add_scalar('Val_memory', np.max(memory_allocated), epoch)

    return np.mean(nmse_losses), np.mean(psnr_losses), np.mean(mse_losses), np.mean(ssim_losses), \
           time.perf_counter() - start, np.max(memory_allocated)


def visualize(args, epoch, model, data_ds, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    output_images = []
    target_images = []
    corrupted_images = []

    mask_c = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)
    mask_c = torch.from_numpy(mask_c[None, None, :, :].astype(np.float32))
    from_space = lambda x: kspace_to_image(x, (2, 3)).real

    with torch.no_grad():
        i = 0
        for data in get_batches(args, data_ds):
            target = data['image']
            target = torch.from_numpy(target.reshape(-1, 1, args.resolution, args.resolution).astype(np.float32))
            y = from_space(get_kspace(target, axes=(2, 3)) * mask_c)[:, :, :, :, None]
            y = y.repeat(1, 1, 1, 1, 2)
            # y = torch.cat((y, torch.zeros_like(y)), dim=-1)
            mask = mask_c[:, :, 1].clone() # mask shape [1, 1, 256]
            mask = mask[None, :, :, :, None].repeat(args.batch_size, 1, 1, 1, 1)
            target = target.reshape((-1, args.resolution, args.resolution))

            y = y.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)

            n_slices = args.n_slices
            # corrupted = transforms.root_sum_of_squares(estimate_to_image(
            #     y[..., n_slices // 2, :, :, :], target.size()[-2:]), 1)
            corrupted = y[:, 0, :, :, 0]
            # if args.n_slices == 1:
            #     mask = mask.squeeze(-4)
            #     y = y[..., n_slices // 2, :, :, :]

            estimate = model.forward(y=y, mask=mask, metadata=None)
            estimate.detach_()

            # target = target[..., n_slices // 2, :, :]
            target_norm = target.norm(dim=(-2, -1), keepdim=True)
            corrupted_images.append(corrupted / target_norm)
            target_images.append(target / target_norm)
            if args.n_slices > 1:
                output_images.append(
                    estimate_to_image(estimate[..., n_slices // 2, :, :, :],
                                      target.size()[-2:]).clone().detach() / target_norm)
            else:
                output_images.append(estimate_to_image(estimate).clone().detach() / target_norm)
                # output_images.append(estimate_to_image(estimate, target.size()[-2:]).clone().detach() / target_norm)
            i+=1
            if i > 3:
                break
    output = torch.cat(output_images, 0)[:16].unsqueeze(1)
    target = torch.cat(target_images, 0)[:16].unsqueeze(1)
    corrupted = torch.cat(corrupted_images, 0)[:16].unsqueeze(1)

    save_image(target, 'Target')
    save_image(corrupted, 'Corrupted')
    save_image(output, 'Reconstruction')
    save_image(target - output, 'Error')
    save_image(corrupted - output, 'Corrupted_Reconstruction_Difference')
    save_image(corrupted - target, 'Corrupted_Target_Difference')


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    im_channels = 2 if args.challenge == 'singlecoil' else 30
    if args.use_rim:
        conv_nd = 3 if args.n_slices > 1 else 2
        rnn = ConvRNN(2 * im_channels, conv_dim=conv_nd)
        model = RIM(rnn, grad_fun=mse_gradient)
        model = RIMfastMRI(model, n_steps=args.n_steps)
    else:
        im_channels = im_channels * args.multiplicity
        channels = args.n_hidden if isinstance(args.n_hidden, list) else [args.n_hidden] * args.depth
        assert channels[0] >= 2 * im_channels + 8
        dilations = args.dilations
        n_hidden = args.n_network_hidden
        conv_nd = 3 if args.n_slices > 1 else 2
        if conv_nd == 3:
            # Make sure to not downsample in the slice direction
            dilations = [[1, d, d] for d in dilations]
        if args.shared_weights:
            cell_ = InvertibleUnet(n_channels=channels, n_hidden=n_hidden, dilations=dilations, conv_nd=conv_nd)
            cell = torch.nn.ModuleList([cell_] * args.n_steps)
        else:
            cell = torch.nn.ModuleList([InvertibleUnet(n_channels=channels, n_hidden=n_hidden,
                                                       dilations=dilations, conv_nd=conv_nd)
                                        for i in range(args.n_steps)])
        if args.parametric_output:
            output_function = ResidualBlockPixelshuffle(channels[0], 2, channels[0], conv_nd=conv_nd, use_glu=False)
        else:
            output_function = lambda x: complex_to_real(real_to_complex(x)[:,:im_channels // (2 * args.multiplicity)])

        model = IRIM(cell, grad_fun=mse_gradient, n_channels=im_channels)
        model = IRIMfastMRI(model, output_function, channels[0], multiplicity=args.multiplicity)

    return model.to(args.device)


def load_model(checkpoint_file):
    device_id = torch.cuda.current_device()
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(device_id))
    args = checkpoint['args']
    model = build_model(args)
    # if args.data_parallel:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    if args.optimizer.upper() == 'RMSPROP':
        optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'ADAM':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'SGD':
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    checkpoint_pretrained = os.path.join(args.exp_dir, 'pretrained.pt')
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.exp_dir, 'model.pt')
    else:
        checkpoint_path = args.checkpoint

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint, model, optimizer = load_model(checkpoint_path)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args).to(device)
        if args.data_parallel:
            # model = torch.nn.DataParallel(model)
            model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        optimizer = build_optim(args, model.parameters())
        if os.path.exists(checkpoint_pretrained):
            _, model, optimizer = load_model(checkpoint_pretrained)
            optimizer.lr = args.lr
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    # train_loader, val_loader, display_loader = create_training_loaders(args)
    train_ds, eval_ds, _ = get_dataset(args, additional_dim=1, uniform_dequantization=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    # if rank <= 0:
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_ds, optimizer, writer)
        scheduler.step(epoch)
        if dist.get_rank() == 0:
            nmse_loss, psnr_loss, mse_loss, ssim_loss, dev_time, dev_mem = evaluate(args, epoch, model, eval_ds, writer)
            visualize(args, epoch, model, eval_ds, writer)
        
            is_new_best = -ssim_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, ssim_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'VAL_NMSE = {nmse_loss:.4g} VAL_MSE = {mse_loss:.4g} VAL_PSNR = {psnr_loss:.4g} '
                f'VAL_SSIM = {ssim_loss:.4g} \n'
                f'TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s ValMemory = {dev_mem:.2f}',
            )
        
        if args.exit_after_checkpoint:
            writer.close()
            sys.exit(0)
    writer.close()


def create_arg_parser():
    parser = Args()
    # Mask parameters
    parser.add_argument('--dataset', type=str, default='brats')
    parser.add_argument('--n_keep', default=32, type=int)
    # parser.add_argument('--resolution', type=int, nargs=2, default=None, help='Image resolution during training')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--evalbatch_size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of slices in an observation. Default=1, if'
                                                                'n_slices > 1, we will use 3d convolutions')
    parser.add_argument('--random_flip', action='store_true')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--n_iters', type=int, default=5000, help='Number of training iterations per epoch')
    parser.add_argument('--use_rim', action='store_true',
                        help='If set, RIM with fixed parameters')
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use choose between"
                                                                      "['Adam', 'SGD', 'RMSProp']")
    parser.add_argument('--loss', choices=['l1', 'mse', 'nmse', 'ssim'], default='ssim', help='Training loss')
    parser.add_argument('--loss_subsample', type=float, default=1., help='Sampling rate for loss mask')
    parser.add_argument('--use_rss', action='store_true',
                        help='If set, will train singlecoil model with RSS targets')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of RIM steps')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--shared_weights', action='store_true',
                        help='If set, weights will be shared over time steps. (only relevant for IRIM)')
    parser.add_argument('--n_hidden', type=int, nargs='+', default=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], 
                        help='Number of hidden features in each layer. Can be either Int or List of Ints')
    parser.add_argument('--n_network_hidden', type=int, nargs='+', default=[64, 64, 128, 128, 256, 1024, 1024, 256, 128, 128, 64, 64],
                        help='Number of hidden features in each layer. Can be either Int or List of Ints')
    parser.add_argument('--dilations', type=int, nargs='+', default=[1, 1, 2, 2, 4, 8, 8, 4, 2, 2, 1, 1],
                        help='Kernel dilations in each in each layer. Can be either Int or List of Ints')
    parser.add_argument('--depth', type=int, help='Number of RNN layers.')
    
    parser.add_argument('--report_interval', type=int, default=200, help='Period of loss reporting')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--multiplicity', type=int, default=4,
                        help='Number of eta estimates at every time step. The higher multiplicity, the lower the '
                             'number of necessary time steps would be expected.')
    # parser.add_argument('--train_resolution', type=int, nargs=2, default=None, help='Image resolution during training')
    parser.add_argument('--parametric_output', action='store_true', help='Use a parametric function for map the'
                                                                         'last layer of the iRIM to an image estimate')
    parser.add_argument('--exit_after_checkpoint', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', type=bool, default=False)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #### distributed training settings ####
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    main(args)
