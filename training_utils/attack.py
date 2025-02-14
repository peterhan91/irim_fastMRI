"""
this code is modified from https://github.com/ylsung/pytorch-adversarial-training
"""
import sys
sys.path.append("..")

import torch
import torch.nn.functional as F

from training_utils.helpers import estimate_to_image


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return 


def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        # dist = F.normalize(dist, p=2, dim=1)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError
    return x


class FastGradientSignUntargeted():
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, mask, epsilon, alpha, min_val=0., max_val=1., max_iters=10, _type='linf'):
        self.model = model
        # self.model.eval()
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        
        if mask is not None:
            self.mask = mask
        
    def perturb(self, original_images, labels, mask, reduction4loss='mean', random_start=False, eval=False):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            y = original_images + rand_perturb
            y.clamp_(self.min_val, self.max_val)
        else:
            y = original_images.clone()
        
        if mask is not None:
            self.mask = mask

        y.requires_grad = True 
        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        self.model.eval()
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(y, self.mask)
                outputs = estimate_to_image(outputs)
                outputs = outputs.reshape(-1, 1, outputs.size(-2), outputs.size(-1))
                labels = labels.reshape_as(outputs)
                # loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
                loss = F.mse_loss(outputs, labels, reduction=reduction4loss)
                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, y, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                y.data += self.alpha * torch.sign(grads.data) 

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                y = project(y, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                y.clamp_(self.min_val, self.max_val)

                if eval:
                    print('Adv. test loss: ', F.mse_loss(y, labels, reduction=reduction4loss).item())

        return y