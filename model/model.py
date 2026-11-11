import logging
from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
import os, glob
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from . import metrics as Metrics
from util.image_pool import ImagePool

class CDARL(BaseModel):
    def __init__(self, opt):
        super(CDARL, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netH = self.set_device(networks.define_F(opt))
        self.netD = self.set_device(networks.define_D(opt))
        self.schedule_phase = None
        self.centered = opt['datasets']['train']['centered']

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        # PatchSampleF creates mlp_0/1/2 on the first forward, not in __init__.
        # A resumed H checkpoint contains those layers, so build them before loading.
        if self.opt['path']['resume_state'] and self.opt['phase'] == 'train':
            self._init_contrastive_head(opt)
        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            self.fake_I_pool = ImagePool(50)
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.load_opt()
            self.log_dict = OrderedDict()
        self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def _init_contrastive_head(self, opt):
        """Build PatchSampleF MLPs before loading a resumed H checkpoint."""
        image_size = opt['model']['diffusion']['image_size']
        channels = opt['model']['diffusion']['channels']
        dummy = {
            'A': torch.zeros(1, channels, image_size, image_size),
            'F': torch.zeros(1, channels, image_size, image_size),
        }
        self.data_dependent_initalize(dummy, opt)

    def data_dependent_initalize(self, data, opt):
        self.data = self.set_device(data)
        output, _ = self.netG(self.data)

        self.A_noisy, self.A_latent, self.B_latent, self.mask_V, self.synt_A, self.mask_F = output

        patchNum = 256
        feat_f, sample_ids = self.netH(self.data['F'], patchNum, None)
        feat_sv, _ = self.netH(self.mask_V, patchNum, sample_ids)
        feat_sf, _ = self.netH(self.mask_F, patchNum, sample_ids)
        
        self.optH = torch.optim.Adam(self.netH.parameters(), lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.netG.loss_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.netG.loss_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        h_alpha = 0.2 
        h_beta = 2.0       

        output, [l_dif, l_cyc] = self.netG(self.data)

        self.A_noisy, self.A_latent, self.B_latent, self.mask_V, self.synt_A, self.mask_F = output 
        l_cyc = l_cyc * h_beta

        self.set_requires_grad(self.netD, True)
        self.optD.zero_grad()  
        synt_A = self.fake_I_pool.query(self.synt_A)
        l_adv_D = self.backward_D_basic(self.netD, self.data['A'], synt_A) * h_alpha
        l_adv_D.backward()
        self.optD.step()

        self.set_requires_grad(self.netD, False)
        self.optG.zero_grad()
        self.optH.zero_grad()
        patchNum = 256
        feat_f, sample_ids = self.netH(self.data['F'], patchNum, None)
        feat_sv, _ = self.netH(self.mask_V, patchNum, sample_ids)
        feat_sf, _ = self.netH(self.mask_F, patchNum, sample_ids)
        l_cont = 0.0
        for f_q, f_sf, f_sv, crit in zip(feat_f, feat_sf, feat_sv, self.netG.loss_nce):
            loss = crit(f_q, f_sf, f_sv) * self.opt['train']['lambda_NCE']
            l_cont += loss.mean()

        l_adv_G = self.netG.loss_gan(self.netD(self.synt_A), True) * h_alpha
        appearance = self._appearance_losses(self.data['A'], self.synt_A, self.data['F'])
        l_tot = l_dif + l_cyc + l_cont + l_adv_G + appearance['total']
        l_tot.backward()
        self.diagnostic_step = getattr(self, 'diagnostic_step', 0) + 1
        freq = self.opt['train'].get('diagnostic_freq', 0)
        if freq and (self.diagnostic_step == 1 or self.diagnostic_step % freq == 0):
            with torch.no_grad():
                grads = [p.grad.norm().item() for p in self.netH.parameters() if p.grad is not None]
                self.log_dict['diag_H_grad_norm_sum'] = sum(grads)
                self.log_dict['diag_feature_positive_negative_mae'] = float((feat_sf[0]-feat_sv[0]).abs().mean())
                self.log_dict['diag_mask_V_std'] = float(self.mask_V.std())
                self.log_dict['diag_mask_F_std'] = float(self.mask_F.std())
                self.log_dict['diag_mask_V_foreground'] = float((self.mask_V>0).float().mean())
                logger.info('Contrastive diagnostics: %s', dict(self.log_dict))
        self.optG.step()
        self.optH.step()

        # set log
        self.log_dict['l_tot'] = l_tot.item()
        self.log_dict['l_dif'] = l_dif.item()
        self.log_dict['l_cyc'] = l_cyc.item()
        self.log_dict['l_cont'] = l_cont.item()
        self.log_dict['l_adv_G'] = l_adv_G.item()
        self.log_dict['l_adv_D'] = l_adv_D.item()
        for key, value in appearance.items():
            self.log_dict['l_' + key] = value.item()

    def _appearance_losses(self, real, synthetic, fractal):
        """Small constraints on vessel contrast, disc rim, brightness, and texture.

        The fractal mask is the only vessel condition available for synt_A.
        Weights are relative to the existing total loss and default to zero.
        """
        weights = self.opt['train'].get('appearance') or {}
        zero = real.new_zeros(())
        losses = {'vessel': zero, 'rim': zero, 'tone': zero, 'texture': zero, 'total': zero}
        if not any(float(weights.get(key, 0)) for key in ('vessel', 'rim', 'tone', 'texture')):
            return losses
        real_01 = (real[:, :1] + 1) * 0.5
        fake_01 = (synthetic[:, :1] + 1) * 0.5
        vessel = fractal[:, :1] > 0
        kernel = torch.ones(1, 1, 11, 11, device=real.device)
        neighbor = torch.nn.functional.conv2d(vessel.float(), kernel, padding=5).clamp(max=1)
        background = neighbor - vessel.float() > 0.5
        if vessel.any() and background.any():
            real_gap = real_01[vessel].mean() - real_01[background].mean()
            fake_gap = fake_01[vessel].mean() - fake_01[background].mean()
            losses['vessel'] = (fake_gap - real_gap.detach()).abs()
        yy = torch.linspace(-1, 1, real.shape[-2], device=real.device)
        xx = torch.linspace(-1, 1, real.shape[-1], device=real.device)
        radius = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        outside = (radius > 0.96).view(1, 1, *radius.shape)
        rim = ((radius > 0.84) & (radius <= 0.96)).view(1, 1, *radius.shape)
        if outside.any():
            losses['rim'] = fake_01[outside].mean()
        if rim.any():
            losses['rim'] = losses['rim'] + (fake_01[rim].mean() - real_01[rim].mean().detach()).abs()
        field = (radius <= 0.96).view(1, 1, *radius.shape)
        nonvessel = field & ~vessel
        if nonvessel.any():
            real_bg = real_01[nonvessel]
            fake_bg = fake_01[nonvessel]
            real_hist = torch.histc(real_bg.detach(), bins=16, min=0, max=1)
            fake_hist = torch.histc(fake_bg, bins=16, min=0, max=1)
            real_hist = real_hist / real_hist.sum().clamp_min(1)
            fake_hist = fake_hist / fake_hist.sum().clamp_min(1)
            losses['tone'] = (fake_hist - real_hist).abs().mean()
            blur = torch.nn.functional.avg_pool2d(fake_01, 7, stride=1, padding=3)
            real_blur = torch.nn.functional.avg_pool2d(real_01, 7, stride=1, padding=3)
            fake_var = (fake_01 - blur).pow(2)
            real_var = (real_01 - real_blur).pow(2)
            losses['texture'] = (fake_var[nonvessel].mean() - real_var[nonvessel].mean().detach()).abs()
        losses['total'] = sum(losses[key] * float(weights.get(key, 0)) for key in ('vessel', 'rim', 'tone', 'texture'))
        return losses

    def test(self):
        self.netG.eval()
        if isinstance(self.netG, nn.DataParallel):
            self.test_V = self.netG.module.segment(self.data, self.opt)
        else:
            self.test_V = self.netG.segment(self.data, self.opt)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, isTrain=True):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        out_dict['dataA'] = Metrics.tensor2im(self.data['A'][0].detach().float().cpu(), min_max=min_max)
        out_dict['dataF'] = Metrics.tensor2im(self.data['F'][0].detach().float().cpu(), min_max=min_max)
        out_dict['A_noisy'] = Metrics.tensor2im(self.A_noisy[0].detach().float().cpu(), min_max=min_max)
        out_dict['A_latent'] = Metrics.tensor2im(self.A_latent[0].detach().float().cpu(), min_max=min_max)
        out_dict['B_latent'] = Metrics.tensor2im(self.B_latent[0].detach().float().cpu(), min_max=min_max)
        out_dict['mask_V'] = Metrics.tensor2im(self.mask_V[0].detach().float().cpu(), min_max=min_max)
        out_dict['synt_A'] = Metrics.tensor2im(self.synt_A[0].detach().float().cpu(), min_max=min_max)
        out_dict['mask_F'] = Metrics.tensor2im(self.mask_F[0].detach().float().cpu(), min_max=min_max)

        if not isTrain:
            out_dict['test_V'] = Metrics.tensor2im(self.test_V[0].detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_segment(self):
        out_dict = OrderedDict()
        out_dict['test_V'] = self.test_V.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info(
            'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, seg_save=False, dice=0):
        if not seg_save:
            G_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_G.pth'.format(iter_step, epoch))
            H_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_H.pth'.format(iter_step, epoch))
            Da_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_Da.pth'.format(iter_step, epoch))
            optG_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_optG.pth'.format(iter_step, epoch))
            optD_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_optD.pth'.format(iter_step, epoch))
            optH_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_optH.pth'.format(iter_step, epoch))
        else:
            segPath = glob.glob(os.path.join(self.opt['path']['checkpoint'], 'D*'))
            for idx in range(len(segPath)):
                os.remove(segPath[idx])
            G_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_G.pth'.format(dice, iter_step, epoch))
            H_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_H.pth'.format(dice, iter_step, epoch))
            Da_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_Da.pth'.format(dice, iter_step, epoch))
            optG_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_optG.pth'.format(dice, iter_step, epoch))
            optD_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_optD.pth'.format(dice, iter_step, epoch))
            optH_path = os.path.join(self.opt['path']['checkpoint'], 'D{}_I{}_E{}_optH.pth'.format(dice, iter_step, epoch))

        # G
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, G_path, _use_new_zipfile_serialization=False)
        # H
        network = self.netH
        if isinstance(self.netH, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, H_path, _use_new_zipfile_serialization=False)
        # # D_a
        network = self.netD
        if isinstance(self.netD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, Da_path, _use_new_zipfile_serialization=False)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, optG_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optD.state_dict()
        torch.save(opt_state, optD_path)
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optH.state_dict()
        torch.save(opt_state, optH_path)
        self._prune_checkpoints()

        logger.info('Saved model in [{:s}] ...'.format(G_path))

    def _prune_checkpoints(self):
        keep = self.opt['train'].get('keep_checkpoints')
        if not keep or keep < 1:
            return
        directory = self.opt['path']['checkpoint']
        groups = {}
        for name in os.listdir(directory):
            if not name.endswith('_G.pth') or name.startswith('D'):
                continue
            prefix = name[:-len('_G.pth')]
            path = os.path.join(directory, name)
            groups[prefix] = os.path.getmtime(path)
        for prefix in sorted(groups, key=groups.get, reverse=True)[keep:]:
            for suffix in ('_G.pth', '_H.pth', '_Da.pth', '_optG.pth', '_optD.pth', '_optH.pth'):
                path = os.path.join(directory, prefix + suffix)
                if os.path.isfile(path):
                    os.remove(path)

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            G_path = '{}_G.pth'.format(load_path)

            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(G_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                H_path = '{}_H.pth'.format(load_path)
                Da_path = '{}_Da.pth'.format(load_path)
                network = self.netH
                if isinstance(self.netH, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(H_path), strict=(not self.opt['model']['finetune_norm']))
                network = self.netD
                if isinstance(self.netD, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(Da_path), strict=(not self.opt['model']['finetune_norm']))

    def load_opt(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            optG_path = '{}_optG.pth'.format(load_path)
            optD_path = '{}_optD.pth'.format(load_path)
            optH_path = '{}_optH.pth'.format(load_path)

            # optimizer
            optG = torch.load(optG_path)
            self.optG.load_state_dict(optG['optimizer'])
            self.begin_step = optG['iter']
            self.begin_epoch = optG['epoch']
            optD = torch.load(optD_path)
            self.optD.load_state_dict(optD['optimizer'])
            self.begin_step = optD['iter']
            self.begin_epoch = optD['epoch']
            optH = torch.load(optH_path)
            self.optH.load_state_dict(optH['optimizer'])
            self.begin_step = optH['iter']
            self.begin_epoch = optH['epoch']
