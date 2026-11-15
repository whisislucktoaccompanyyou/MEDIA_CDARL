import math
import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import loss

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, segment_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        refinement=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.segment_fn = segment_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.refinement_opt = refinement or {}
        self.refiner = None
        if self.refinement_opt.get('enabled'):
            from model.refinement import StructureRefiner
            if getattr(segment_fn, 'label_nc', 2) != 2 or channels != 1:
                raise ValueError('refinement requires grayscale and the original 2-channel Gs')
            self.refiner = StructureRefiner(
                channels=self.refinement_opt.get('channels', 32),
                steps=self.refinement_opt.get('steps', 3),
                alpha=self.refinement_opt.get('alpha', 0.1))
        self.path_geometry_weight = 0.0
        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_nce = []
        for i in range(3):
            self.loss_nce.append(loss.MaskPatchNCELoss().to(device))
        
        self.loss_gan = loss.GANLoss('lsgan').to(device)
        self.loss_cyc = torch.nn.L1Loss()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_segment(self, x_in, opt):
        device = self.betas.device
        x_start_ = x_in['A']
        segm_V = torch.zeros_like(x_start_)
        dsize = x_start_.shape[-1]

        if opt['phase'] != 'train':
            if 'STARE' in opt['datasets']['test']['dataroot'] or 'DRIVE' in opt['datasets']['test']['dataroot']:
                for opt1 in range(0,dsize,256):
                    for opt2 in range(0,dsize,256):
                        x_start = x_start_[:, :, opt1:opt1+256, opt2:opt2+256]
                        b= x_start.shape[0]
                        t = torch.full((b,), 0, device=device, dtype=torch.long)
                        x_latent = self.denoise_fn(x_start, t)
                        segm_V[:, :, opt1:opt1+256, opt2:opt2+256] = self.segment_fn(torch.cat([x_start, x_latent], dim=1))
                return segm_V
            
        for opt1 in range(2):
            for opt2 in range(2):
                x_start = x_start_[:, :, opt1::2, opt2::2]
                b= x_start.shape[0]
                t = torch.full((b,), 0, device=device, dtype=torch.long)
                x_latent = self.denoise_fn(x_start, t)
                segm_V[:, :, opt1::2, opt2::2] = self.segment_fn(torch.cat([x_start, x_latent], dim=1))
        return segm_V

    @torch.no_grad()
    def sample(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def segment(self, x_in, opt):
        return self.p_sample_segment(x_in, opt)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def synthesize(self, reference, target, timestep, noise=None, geometry=None,
                   condition=None, refine_steps=None):
        """Single B entry shared by training, monitoring and inference.

        target: centered single-channel vessel mask; timestep: diffusion t, not R's k.
        """
        if self.refiner is not None and condition is not None:
            raise ValueError('legacy C is forbidden in refinement mode; geometry must derive from F')
        if isinstance(timestep, int):
            timestep = torch.full((reference.shape[0],), timestep, device=reference.device, dtype=torch.long)
        if ((timestep < 0) | (timestep >= self.num_timesteps)).any():
            raise ValueError('diffusion timestep outside schedule')
        noisy = self.q_sample(reference, timestep, noise=noise)
        latent = self.denoise_fn(noisy, timestep)
        if condition is None:
            classes = (target[:, 0] > 0).long()
            condition = torch.nn.functional.one_hot(classes, 2).permute(0, 3, 1, 2).to(reference)
        initial = self.segment_fn(torch.cat([noisy, latent], 1), condition.to(reference))
        if self.refiner is not None:
            if geometry is None:
                from data.geometry import batch_geometry
                geometry = batch_geometry(target)
            images = self.refiner(initial, reference, geometry, steps=refine_steps)
        else:
            images = [initial]
        return {'initial': initial, 'images': images, 'final': images[-1],
                'noisy': noisy, 'latent': latent, 'geometry': geometry}

    def cycle(self, image, timestep, noise=None):
        noisy = self.q_sample(image, timestep, noise=noise)
        return self.segment_fn(torch.cat([noisy, self.denoise_fn(noisy, timestep)], 1))

    def p_losses(self, x_in, noise=None):
        a_start = x_in['A']
        b = a_start.shape[0]
        device = a_start.device
        noise = default(noise, lambda: torch.randn_like(a_start))
        t_a = torch.randint(0, min(200, self.num_timesteps), (b,), device=device).long()
        A_noisy = self.q_sample(a_start, t_a, noise=noise)
        A_latent = self.denoise_fn(A_noisy, t_a)
        mask_V = self.segment_fn(torch.cat([A_noisy, A_latent], 1))
        t_b = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        result = self.synthesize(a_start, x_in['F'], t_b, noise=noise,
                                 geometry=x_in.get('Cf'), condition=x_in.get('C'))
        synt_A = result['final']
        mask_F = self.cycle(synt_A, t_a, noise=noise)
        l_dif = self.loss_func(noise, result['latent']) / noise.numel()
        l_cyc = self.loss_cyc(mask_F, x_in['F'])
        output = [A_noisy, A_latent, result['latent'], mask_V, synt_A, mask_F]
        if self.refiner is None:
            return output, [l_dif, l_cyc]
        from .loss import geometry_losses
        final_dice, final_boundary = geometry_losses(mask_F, x_in['F'], x_in.get('V'))
        path_dice = path_boundary = synt_A.new_zeros(())
        path_index = 0
        # Pick one *refined*, non-final step; I0 is retained as a baseline.
        if self.path_geometry_weight > 0 and len(result['images']) > 2:
            path_index = int(torch.randint(1, len(result['images']) - 1, ()).item())
            prediction = self.cycle(result['images'][path_index], t_a, noise=noise)
            path_dice, path_boundary = geometry_losses(prediction, x_in['F'], x_in.get('V'))
        extra = {'synthesis': result, 'dice': final_dice, 'boundary': final_boundary,
                 'path_dice': path_dice, 'path_boundary': path_boundary,
                 'path_index': path_index}
        return output, [l_dif, l_cyc], extra

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

