import os
import glob
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity

import torch

from train import build_model
from utils import get_cartesian_mask, get_kspace, kspace_to_image

from training_utils.helpers import estimate_to_image
from training_utils.attack import FastGradientSignUntargeted


def load_model(checkpoint_file, ddp=False):
    if not ddp:
        checkpoint = torch.load(checkpoint_file)
    else:
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(device_id))
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    return checkpoint, model


def evaluate(_input, _target):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        _target.transpose(1, 2, 0), _input.transpose(1, 2, 0), multichannel=True, data_range=_target.max()
    )


def test(args, model, attack_obj, array, mask, adv_test=True, 
            save_img=False, eval_dir=None, verbose=False):
    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0
    advs = []
    outs = []
    outs_a = []
    mask_c = torch.from_numpy(mask[None, None, :, :].astype(np.float32)).cuda()
    from_space = lambda x: kspace_to_image(x, (2, 3)).real
    
    with torch.no_grad():
        for n, target in tqdm(enumerate(array)):
            target = torch.from_numpy(target.reshape(-1, 1, args.resolution, args.resolution).astype(np.float32))  
            target = target.to(args.device)
            # target_norm = target.norm(dim=(-2, -1), keepdim=True)
            y = from_space(get_kspace(target, axes=(2, 3)) * mask_c)[:, :, :, :, None]
            y = y.repeat(1, 1, 1, 1, 2)
            mask = mask_c[:, :, 1].detach() # mask shape [1, 1, 256]
            mask = mask[None, :, :, :, None].repeat(args.batch_size, 1, 1, 1, 1)
            y = y.to(args.device)
            mask = mask.to(args.device)
            
            model.eval()
            output = model.forward(y=y, mask=mask, metadata=None)
            output.detach_()
            output = estimate_to_image(output.to('cpu'))
            outs.append(output)

            y_s = output.reshape(-1, args.resolution, args.resolution)
            y_t = target.reshape_as(y_s).to('cpu').numpy()
            y_s = y_s.numpy()
            te_acc = evaluate(y_s, y_t)
            # print('standard SSIM: ', te_acc)
            if te_acc < 1:
                total_acc += te_acc
                num += output.shape[0]

            if adv_test == True:
                # use predicted label as target label
                with torch.enable_grad():
                    adv_data = attack_obj.perturb(y, target, mask, 'mean', False)
                    advs.append(adv_data.to('cpu'))
                
                adv_output = model.forward(y=adv_data, mask=mask, metadata=None)
                adv_output = estimate_to_image(adv_output.detach_().to('cpu'))
                outs_a.append(adv_output)
                # y_a = np.clip(adv_output.squeeze().cpu().numpy(), 0, 1)
                y_a = adv_output.reshape(-1, args.resolution, args.resolution).numpy()
                adv_acc = evaluate(y_a, y_t)
                total_adv_acc += adv_acc
            else:
                total_adv_acc = -num
            
            if verbose == True:
                print('standard SSIM: ', te_acc, 'adversarial SSIM: ', adv_acc)

            if save_img:
                def save_image(image):
                    image -= image.min()
                    image /= image.max()
                    return image
                
                y_t = (save_image(y_t)*255.).astype(np.uint8).squeeze()
                # print(label_norm)
                y_a = (save_image(y_a)*255.).astype(np.uint8).squeeze()
                y_s = (save_image(y_s)*255.).astype(np.uint8).squeeze()

                adv_data = (save_image(estimate_to_image(adv_data).squeeze().cpu().numpy())*255.).astype(np.uint8)

                total_u = np.hstack((y_t, y_s, y_a))
                total_d = np.hstack((y_t, y_t, adv_data))
                total = np.vstack((total_u, total_d))
                Path(os.path.join(eval_dir, 'total')).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(os.path.join(eval_dir, 'total', str(n)+'.jpg'), total.squeeze())
            
            torch.cuda.empty_cache()
        
        outs = torch.cat(outs, 0)
        if adv_test:
            advs = torch.cat(advs, 0)
            outs_a = torch.cat(outs_a, 0)
            return total_acc / num , total_adv_acc / num, advs.reshape(-1, 256, 256).numpy(), outs.reshape(-1, 256, 256).numpy().clip(0, 1), outs_a.reshape(-1, 256, 256).numpy().clip(0, 1)

    return total_acc / num , outs.reshape(-1, 256, 256).numpy().clip(0, 1)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='adversarial eval')
    # data setting 
    parser.add_argument('--workdir', default='./eval_irim')
    parser.add_argument('--npy_root', default='../Alzheimer/vol_test_/*')
    parser.add_argument('--resolution', type=int, default=256, help='input resolution')
    parser.add_argument('--n_keep', type=int, default=32, help='number of k-space lines sampled')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # model setting
    parser.add_argument('--checkpoint_path', default='./checkpoints/irim/best_model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    # adversarial attack setting
    parser.add_argument('--adv_test', default=False)
    parser.add_argument('--epsilon', '-e', type=float, default=0.01, 
        help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--alpha', '-a', type=float, default=0.005, 
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', '-k', type=int, default=10, 
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
    
    return parser.parse_args()



if __name__ == '__main__':
    args = create_arg_parser()
    test_files = glob.glob(os.path.join(args.npy_root, '*.npz'))
    
    checkpoint, model = load_model(args.checkpoint_path, ddp=False)
    args_irim = checkpoint['args']

    mask = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)
    attack = FastGradientSignUntargeted(model, 
                                    None,
                                    args.epsilon, 
                                    args.alpha, 
                                    max_iters=args.k, 
                                    _type=args.perturbation_type)
    
    print(args.n_keep, args.adv_test)
    for file in test_files:
        R = int(256 / args.n_keep)
        eval_dir = os.path.join(args.workdir, os.path.basename(file).split('.')[0], f'R_{R}')
        Path(eval_dir).mkdir(parents=True, exist_ok=True)

        if file.endswith('.npz'):
            test_imgs = np.load(file)['label']
        elif file.endswith('.npy'):
            test_imgs = np.load(file)

        if args.adv_test == True:
            print('running with adversarial attack!')
            with torch.no_grad():
                acc, acc_adv, aarr, out_arr, out_aarr = test(args, model, attack, test_imgs, mask,
                                    save_img=False, eval_dir=eval_dir, adv_test=True, verbose=False)
            Path(os.path.join(args.workdir, 'adv')).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(args.workdir, 'adv', os.path.basename(file).split('.')[0]+'.npy'), aarr)
            np.savez_compressed(os.path.join(eval_dir, "recons.npz"), 
                                                        recon = out_arr,
                                                        # adv_in = aarr,
                                                        adv_recon = out_aarr)
        else:
            with torch.no_grad():
                acc, out_arr = test(args, model, attack, test_imgs, mask,
                                    save_img=False, eval_dir=eval_dir, adv_test=False, verbose=False)
                print('standard SSIM: ', acc)
            np.savez_compressed(os.path.join(eval_dir, "recons.npz"), 
                                                        recon = out_arr)            


