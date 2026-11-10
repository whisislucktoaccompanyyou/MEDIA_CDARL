import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import numpy as np
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(inference)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    batchSize = opt['datasets']['train']['batch_size']
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset_xcad(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
            val_set = Data.create_dataset_xcad(dataset_opt, 'val')
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
            valid_iters = int(ceil(val_set.data_len / float(batchSize))) if val_set.data_len else 0
        if phase == 'test':
            val_set = Data.create_dataset_xcad(dataset_opt, 'test')
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            valid_iters = int(ceil(val_set.data_len))
    logger.info('Initial Dataset Finished')

    # Reproducible model initialization and sampling.
    import random
    seed = opt['train'].get('seed', 2026)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    val_dice = 0
    if opt['phase'] == 'train':
        max_steps = opt['train'].get('max_steps') or float('inf')
        while current_epoch < n_epoch and current_step < max_steps:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1
                if current_epoch == 1 and istep == 0:
                    diffusion.data_dependent_initalize(train_data, opt)
                    continue
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                save_freq = opt['train'].get('save_checkpoint_freq')
                if save_freq and current_step % save_freq == 0:
                    diffusion.save_network(current_epoch, current_step)
                synth = opt.get('synthesis')
                if synth and (current_step == 2 or current_step % synth['frequency'] == 0):
                    from synthesis_validation import validate
                    net = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
                    validate(net, synth['input_dir'], os.path.join(opt['path']['results'], 'step_%07d' % current_step), synth['timesteps'], synth['seed'])
                    logger.info('Saved fixed-condition synthesis at step %d', current_step)
                # log
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results_step(visuals)

                # validation
                if (current_step+1) % opt['train']['val_freq'] == 0:
                    diffusion.test()
                    visuals = diffusion.get_current_visuals(isTrain=False)
                    visualizer.display_current_results_step(visuals)

                if current_step >= max_steps:
                    break

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)


            dice_per_case_score = 0
            if valid_iters == 0:
                logger.info('No labeled validation split; skip Dice and keep training.')
                continue
            for idata, val_data in enumerate(val_loader):
                diffusion.feed_data(val_data)
                diffusion.test()
                visuals = diffusion.get_current_segment()
                predseg = visuals['test_V'].squeeze().numpy()
                predseg = (predseg + 1) / 2.
                predseg = (predseg > 0.5).astype(bool)

                label = val_data['F'].cpu().squeeze().numpy()
                label = (label + 1) / 2.
                label = (label > 0.5).astype(bool)
                dice = (visualizer.calculate_score(label, predseg, "dice"))
                dice_per_case_score += dice
            dice_case = (dice_per_case_score) / valid_iters
            if dice_case >= val_dice:
                val_dice = dice_case
                diffusion.save_network(current_epoch, current_step, seg_save=True, dice=round(val_dice, 4))

        # The periodic save only fires on epochs divisible by save_checkpoint_epoch.
        # Always keep the last fundus weights as well.
        logger.info('Saving final models and training states.')
        diffusion.save_network(current_epoch, current_step)
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        file = open(result_path + 'test_score.txt', 'w')
        dice_per_case_score = []
        prec_per_case_score = []
        jacc_per_case_score = []
        for idata,  val_data in enumerate(val_loader):
            dataInfo = val_data['P'][0].split('\\')[-1][:-4]
            time1 = time.time()
            diffusion.feed_data(val_data)
            diffusion.test()
            time2 = time.time()

            visuals = diffusion.get_current_segment()
            predseg = visuals['test_V'].squeeze().numpy()
            predseg = (predseg + 1) / 2.
            predseg = (predseg > 0.5).astype(bool)

            label = val_data['F'].cpu().squeeze().numpy()
            label = (label + 1) / 2.
            label = (label > 0.5).astype(bool)

            data = val_data['A'].cpu().squeeze().numpy()
            data = (data+1)/2.
            savePath = os.path.join(result_path, '%d_data.png' % idata)
            save_image(data * 255, savePath)
            savePath = os.path.join(result_path, '%d_pred.png' % (idata))
            save_image(predseg * 255, savePath)
            savePath = os.path.join(result_path, '%d_label.png' % (idata))
            save_image(label * 255, savePath)

            dice = (visualizer.calculate_score(label, predseg, "dice"))
            prec = (visualizer.calculate_score(label, predseg, "prec"))
            jacc = (visualizer.calculate_score(label, predseg, "jacc"))
            file.write('%04d: process image... %03s | Dice=%f | Prec=%f | Jacc=%f \n' % (idata, dataInfo, dice, prec, jacc))

            print('%04d: process image... %s' % (idata, dataInfo))
            dice_per_case_score.append(dice)
            prec_per_case_score.append(prec)
            jacc_per_case_score.append(jacc)

        print("score_dice_per_case: %.04f +- %.03f" % (np.mean(dice_per_case_score), np.std(dice_per_case_score)))
        print("score_prec_per_case : %.04f +- %.03f" % (np.mean(prec_per_case_score), np.std(prec_per_case_score)))
        print("score_jacc_per_case : %.04f +- %.03f" % (np.mean(jacc_per_case_score), np.std(jacc_per_case_score)))
        file.write('score_case | Dice=%f | Precision=%f | Jaccard=%f \n'
                   % (np.mean(dice_per_case_score), np.mean(prec_per_case_score), np.mean(jacc_per_case_score)))
        file.write('S_std_case | Dice=%f | Precision=%f | Jaccard=%f \n'
                   % (np.std(dice_per_case_score), np.std(prec_per_case_score), np.std(jacc_per_case_score)))
        file.close()