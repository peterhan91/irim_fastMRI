import os
import glob
import argparse
from xmlrpc.client import Boolean
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity

import torch

from scripts.train_resunet import load_model
from utils import get_cartesian_mask
from training_utils.attack import FastGradientSignUntargeted


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
    with torch.no_grad():
        for n, label in tqdm(enumerate(array)):
            label = torch.from_numpy(label.reshape(-1, 1, args.resolution, args.resolution).astype(np.float32))  
            label = label.to(args.device)
            model.eval()
            mask = mask_c.detach()
            output = model(label, mask)
            outs.append(output.to('cpu'))

            y_s = output.reshape(-1, args.resolution, args.resolution).to('cpu')
            y = label.reshape_as(y_s).to('cpu').numpy()
            y_s = y_s.numpy()
            te_acc = evaluate(y_s, y)
            # print('standard SSIM: ', te_acc)
            if te_acc < 1:
                total_acc += te_acc
                num += output.shape[0]
            # print(adv_test, args.adv_test)
            if adv_test == True:
                # use predicted label as target label
                with torch.enable_grad():
                    adv_data = attack_obj.perturb(label, label, 'mean', False)
                    advs.append(adv_data.to('cpu'))
                
                adv_output = model(adv_data, mask)
                outs_a.append(adv_output.to('cpu'))
                # y_a = np.clip(adv_output.squeeze().cpu().numpy(), 0, 1)
                y_a = adv_output.reshape(-1, args.resolution, args.resolution).to('cpu').numpy()
                adv_acc = evaluate(y_a, y)
                total_adv_acc += adv_acc
            else:
                total_adv_acc = -num
            
            if verbose:
                print('standard SSIM: ', te_acc, 'adversarial SSIM: ', adv_acc)

            if save_img:
                def save_image(image):
                    image -= image.min()
                    image /= image.max()
                    return image
                
                y = (save_image(y)*255.).astype(np.uint8).squeeze()
                # print(label_norm)
                y_a = (save_image(y_a)*255.).astype(np.uint8).squeeze()
                y_s = (save_image(y_s)*255.).astype(np.uint8).squeeze()

                adv_data = (save_image(adv_data.squeeze().cpu().numpy())*255.).astype(np.uint8)

                total_u = np.hstack((y, y_s, y_a))
                total_d = np.hstack((y, y, adv_data))
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
    parser.add_argument('--workdir', default='./eval_unet')
    parser.add_argument('--npy_root', default='../Alzheimer/vol_test_/*/')
    parser.add_argument('--resolution', type=int, default=256, help='input resolution')
    parser.add_argument('--n_keep', type=int, default=32, help='number of k-space lines sampled')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # model setting
    parser.add_argument('--checkpoint_path', default='./checkpoints/unet/best_model.pt')
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
    
    _, model, _ = load_model(args.checkpoint_path)
    mask = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)
    attack = FastGradientSignUntargeted(model, 
                                    torch.from_numpy(mask[None, None, :, :].astype(np.float32)).cuda(),
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
                acc, acc_adv, aarr, out_arr, out_aarr = test(args, model, attack, test_imgs, mask, adv_test=True,
                                    save_img=False, eval_dir=eval_dir, verbose=False)
            Path(os.path.join(args.workdir, 'adv')).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(args.workdir, 'adv', os.path.basename(file).split('.')[0]+'.npy'), aarr)
            np.savez_compressed(os.path.join(eval_dir, "recons.npz"), 
                                                        recon = out_arr,
                                                        # adv_in = aarr,
                                                        adv_recon = out_aarr)
        else:
            with torch.no_grad():
                acc, out_arr = test(args, model, attack, test_imgs, mask, adv_test=False,
                                    save_img=False, eval_dir=eval_dir, verbose=False)
                print('standard SSIM: ', acc)
            np.savez_compressed(os.path.join(eval_dir, "recons.npz"), 
                                                        recon = out_arr)   



