import os
import piq
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils import get_cartesian_mask, get_kspace, kspace_to_image
from training_utils.helpers import estimate_to_image
from scripts.train_resunet import load_model
from utils import get_cartesian_mask
from train import build_model

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu = 0
    torch.cuda.set_device(gpu)
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor


class rMSELoss(torch.nn.Module):
    def __init__(self):
        super(rMSELoss,self).__init__()

    def forward(self, net_input, x, y):
        criterion = nn.MSELoss()
        loss = -criterion(x,y)
        return loss


def norm(vol):
    vol_tmp = vol - vol.min()
    return vol_tmp / vol_tmp.max()


def load_model_irim(checkpoint_file, ddp=False):
    if not ddp:
        checkpoint = torch.load(checkpoint_file)
    else:
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(device_id))
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    return checkpoint, model


def get_metric(predictions, targets):
    return (piq.psnr(predictions, targets, data_range=1.).item(),
            piq.ssim(predictions, targets, data_range=1.).item())


def runner(net, # network model
            label, # ground truth image; new input
            eps = 0.04,
            num_iter = 5000,
            LR = 0.01,
            mask = None,
            devices = [torch.device("cuda:0")],
            weight_decay=0,
            model_type='unet',
            retain_graph = False,
            require_grad = True
        ):
    # set model grads to false
    if not require_grad:
        for param in net.parameters():
            param.requires_grad = False
    ###################### fixed reconstruction from non-perturbed data ######################   
    mask = torch.from_numpy(mask[None, None, :, :].astype(np.float32)).cuda().detach() # shape [1,1,256,256]
    y = torch.from_numpy(label.reshape(-1, 1, 256, 256).astype(np.float32)).to('cuda') # shape [1,1,256,256]
    ksp = get_kspace(y, axes=(2, 3)).clone() # shape [1,1,256,256]
    masked_ksp = ksp * mask # shape [1,1,256,256]
    inp2 = masked_ksp.to('cuda')
    net.eval()

    if model_type == 'unet':
        out2 = net(ksp_y=inp2, mask=mask)

    elif model_type == 'irim':
        output = net.forward(ksp_y=inp2, mask_c=mask, metadata=None)
        out2 = estimate_to_image(output)    
    # torch.cuda.empty_cache()

    ###################### prepare adversarial variable ###################### 
    shape = ksp.shape
    # print("input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype).to(devices[0])
    net_input.data.normal_()
    net_input.data *= torch.norm(ksp)/torch.norm(net_input)/10000    #1./1e3
    net_input = net_input.type(dtype).to(devices[0])
    net_input.requires_grad = True
    p = [net_input]

    optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    mse = rMSELoss()
    mse_ = np.zeros(num_iter) # list of zeros

    ###################### reconstruction from perturbed data ######################
    pert_recs = []
    R = []
    norm_ratios = []
    ctr = 0
    
    for i in range(num_iter):
        inp = (net_input + ksp) * mask # perturb full k-space and subsampling
        inp = inp.to('cuda') # masked ksp
        if model_type == 'unet':
            out = net(ksp_y=inp, mask=mask)
        
        elif model_type == 'irim':
            output = net.forward(ksp_y=inp, mask_c=mask, metadata=None)
            out = estimate_to_image(output) 
        pert_recs.append(out.data.cpu().numpy()[0])
        
        def closure():
            optimizer.zero_grad()
            loss = mse(net_input.to(devices[-1]), out, out2)
            loss.backward(retain_graph=retain_graph)
            mse_[i] = loss.data.cpu().numpy()
            if i % 10 == 0:
                print ('Iteration %05d   loss: %f norm ration: %03f' % (i, mse_[i], 
                                                                        torch.norm(net_input) / torch.norm(ksp)), '\r', end='')
            return loss   
        
        loss = optimizer.step(closure)
        ### projection onto the l2-norm ball denoted by eps 
        if torch.norm(net_input.data) / torch.norm(ksp) > eps:
            net_input.data = net_input.data * torch.norm(ksp) / (torch.norm(net_input.data) + 1e-10) # 1e-10 is a stabilizer
            net_input.data = net_input.data * eps
            ctr += 1
        R.append(net_input.data.cpu())
        norm_ratios = [torch.norm(net_input)/torch.norm(ksp)] + norm_ratios
        ### ending constraints
        if ctr > 100: # if more than 100 projections have been done
            break
        if len(norm_ratios) >= 200: # if the perturbation norm doesn't change more than 0.1 over 200 iterations
            if norm_ratios[0] - norm_ratios[-1] < 0.01:
                break
            else:
                norm_ratios.pop()
        if eps < 1e-10:
            break
    
    return R, mse_, out2.data.cpu().numpy()[0], pert_recs



if __name__ == '__main__':
    class args():
        def __init__(self):
            self.npy_root = '../Alzheimer/testvol_adv/'
            self.checkpoint_path = './checkpoints/irim/best_model.pt'
            self.resolution = 256
            self.n_keep = 32 # R = 256 / 32 = 8
            self.irim = True
            self.save_root = './eval_irim/adv' if self.irim else './eval_unet/adv'
    args = args()

    file = glob.glob(os.path.join(args.npy_root, 'file.npy'))[0]
    if args.irim:
        _, model = load_model_irim(args.checkpoint_path, ddp=False)
        _, net_, _ = load_model('./checkpoints/unet/best_model.pt')
    else:
        _, model, _ = load_model(args.checkpoint_path)
        _, net_ = load_model_irim('./checkpoints/irim/best_model.pt', ddp=False)

    mask = get_cartesian_mask([args.resolution, args.resolution], args.n_keep)

    if file.endswith('.npy'):
        test_imgs = np.load(file)
    es = np.arange(0, 9, dtype=int) * 0.01
    
    data_psnr_r, data_ssim_r = defaultdict(list), defaultdict(list)
    psnrs_c, ssims_c = [], []

    # iterate through all images and all eps
    for m, e in enumerate(es):
        psnrs_a, ssims_a = [], []
        psnrs_another, ssims_another = [], []
        vol_adv = []
        recon_curr, recon_tran = [], []
        for n, orig in enumerate(tqdm(test_imgs)):       
            ### perform attacks     
            R, loss, clean_rec, pert_recs = runner(net = model,
                                                label=orig,
                                                eps = e,
                                                num_iter = 2000, 
                                                LR = 0.1,
                                                model_type = 'irim' if args.irim else 'unet',
                                                mask = mask,
                                                weight_decay=0,
                                                retain_graph = True
                                            )
            
            ########### pick the worst perturbation across the iterations ###########
            arg = loss.argmin() # worst perturb index
            R_adv = R[arg].cuda()
            net_.eval()
            mask_cuda = torch.from_numpy(mask[None, None, :, :].astype(np.float32)).cuda().detach() # shape [1,1,256,256]
            y = torch.from_numpy(orig.reshape(-1, 1, 256, 256).astype(np.float32)).to('cuda') # shape [1,1,256,256]
            ksp_cuda = get_kspace(y, axes=(2, 3)).clone() # shape [1,1,256,256]

            inp = (R_adv + ksp_cuda) * mask_cuda # perturb full k-space and subsampling
            inp = inp.to('cuda') # masked ksp
            if args.irim:
                out = net_(ksp_y=inp, mask=mask_cuda)
            else:
                output = net_.forward(ksp_y=inp, mask_c=mask_cuda, metadata=None)
                out = estimate_to_image(output) 
            recon_tran.append(norm(out.data.squeeze().cpu().numpy()))
            recon_curr.append(norm(np.squeeze(pert_recs[arg])))
            
            ########### transfer attack and evaluate ###########
            im2 = out.data.cpu().numpy()[0]
            im2 = np.squeeze(norm(im2.copy()))
            im1 = np.squeeze(norm(orig.copy()))
            im1_tensor = torch.from_numpy(im1[None, None, ...]).double()
            im2_tensor = torch.from_numpy(im2[None, None, ...]).double()
            psnr_another, ssim_another = get_metric(im1_tensor, im2_tensor)
            psnrs_another.append(psnr_another)
            ssims_another.append(ssim_another)
            
            ########### save the perturbed volume ###########
            ksp_adv = R_adv + ksp_cuda
            from_space = lambda x: kspace_to_image(x, (2, 3)).real
            vol_adv.append(norm(from_space(ksp_adv).data.squeeze().cpu().numpy()))
    
        ########### get delta psnr, ssim ###########
            P, S = [], []
            for j, im in enumerate([clean_rec] + pert_recs):
                im1 = np.squeeze(norm(orig.copy()))
                im2 = np.squeeze(norm(im.copy()))
                im1_tensor = torch.from_numpy(im1[None, None, ...]).double()
                im2_tensor = torch.from_numpy(im2[None, None, ...]).double()
                if j == 0:
                    psnr_c, ssim_c = get_metric(im1_tensor, im2_tensor)
                if j > 0:
                    psnr_, ssim_ = get_metric(im1_tensor, im2_tensor)
                    P.append(psnr_)
                    S.append(ssim_)            
            psnr_a, ssim_a = P[arg], S[arg]
            psnrs_a.append(psnr_a)
            ssims_a.append(ssim_a)
            if m==0:
                psnrs_c.append(psnr_c)
                ssims_c.append(ssim_c)
        data_psnr_r['%.2f' % e], data_ssim_r['%.2f' % e] = psnrs_a, ssims_a
        data_psnr_r['%.2f' % e + '_an'], data_ssim_r['%.2f' % e + '_an'] = psnrs_another, ssims_another
        np.save(os.path.join(args.save_root, 'perb_%.2f.npy' % e), np.array(vol_adv))
        np.save(os.path.join(args.save_root, 'recon_current_%.2f.npy' % e), np.array(recon_curr))
        np.save(os.path.join(args.save_root, 'recon_transfer_%.2f.npy' % e), np.array(recon_tran))
        
    assert len(psnrs_c) == len(test_imgs)
    data_psnr_r['clean'], data_ssim_r['clean'] = psnrs_c, ssims_c

    df_psnr = pd.DataFrame(data=data_psnr_r)
    df_ssim = pd.DataFrame(data=data_ssim_r)
    df_psnr.to_csv(os.path.join(args.save_root, 'psnr_adv.csv'), index=False)
    df_ssim.to_csv(os.path.join(args.save_root, 'ssim_adv.csv'), index=False)

