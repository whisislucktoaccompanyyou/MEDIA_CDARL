'''create dataset and dataloader'''
import logging
import torch.utils.data
from torch.utils.data import WeightedRandomSampler

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        sampler = None
        shuffle = dataset_opt['use_shuffle']
        if getattr(dataset, 'sample_weights', None):
            sampler = WeightedRandomSampler(
                dataset.sample_weights, len(dataset.sample_weights), replacement=True)
            shuffle = False
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        # raise NotImplementedError('Dataloader [{:s}] is not found.'.format(phase))

def create_dataset_xcad(dataset_opt, phase):
    '''create dataset'''
    from data.XCAD_dataset import XCADDataset as D
    dataset = D(
        dataroot=dataset_opt['dataroot'],
        split=phase,
        source_dir=dataset_opt.get('source_dir'),
        quality_csv=dataset_opt.get('quality_csv'),
        quality_weights=dataset_opt.get('quality_weights'),
        condition_dir=dataset_opt.get('condition_dir'),
        image_size=dataset_opt.get('image_size', 256),
    )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset