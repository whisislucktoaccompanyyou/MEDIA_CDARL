"""Deterministic vessel-conditioned synthesis; never modifies training RNG state."""
import os, json
import cv2
import numpy as np
import torch
from test_synthesis import find_pairs, read_gray, to_model_tensor, tensor_to_u8, panel, save_gray

def mask_classes(mask, size=256, threshold=0.5):
    values = mask.astype(np.float32)
    if values.max() > 1:
        values /= 255.0
    binary = (values >= threshold).astype(np.uint8)
    return cv2.resize(binary, (size,size), interpolation=cv2.INTER_NEAREST)

def validate(net, input_dir, output_dir, timesteps, seed=2026):
    os.makedirs(output_dir, exist_ok=True)
    pairs, missing = find_pairs(input_dir)
    if not pairs or missing:
        raise RuntimeError('Missing synthesis image/mask pairs: '+str(missing))
    device = next(net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    training = net.training
    net.eval()
    report=[]
    try:
        with torch.no_grad():
            for stem, ip, mp in pairs:
                original, mask = read_gray(ip), read_gray(mp)
                if original.shape != mask.shape:
                    raise ValueError('Image/mask shape mismatch: '+stem)
                image = to_model_tensor(original).to(device)
                classes = torch.from_numpy(mask_classes(mask)).long().unsqueeze(0).to(device)
                condition = torch.nn.functional.one_hot(classes,2).permute(0,3,1,2).float()
                noise = torch.randn(image.shape, generator=generator, device=device)
                for t in timesteps:
                    if not 0 <= t < net.num_timesteps:
                        raise ValueError('Invalid timestep')
                    timestep = torch.full((1,),t,device=device,dtype=torch.long)
                    noisy=net.q_sample(image,timestep,noise=noise)
                    latent=net.denoise_fn(noisy,timestep)
                    features=torch.cat([noisy,latent],1)
                    synthetic=net.segment_fn(features,condition)
                    # Identical reference/noise, changed condition: measures whether mask is used.
                    changed=net.segment_fn(features,torch.flip(condition,[3]))
                    reconstructed=net.segment_fn(torch.cat([synthetic,net.denoise_fn(synthetic,torch.zeros_like(timestep))],1))
                    pred=reconstructed[:,0]>0
                    target=classes.bool()
                    dice=(2*(pred & target).sum().float()+1e-6)/(pred.sum()+target.sum()+1e-6)
                    u8=tensor_to_u8(synthetic)
                    prefix=os.path.join(output_dir,stem+'_t'+str(t))
                    save_gray(prefix+'_synthetic.png',u8)
                    save_gray(prefix+'_panel.png',panel(original,mask_classes(mask)*255,u8))
                    report.append(dict(id=stem,timestep=t,seed=seed,mask_fraction=float(target.float().mean()),synthetic_mean=float(u8.mean()),condition_change_mae=float((synthetic-changed).abs().mean()),cycle_dice=float(dice)))
    finally:
        net.train(training)
    with open(os.path.join(output_dir,'synthesis_report.json'),'w') as f:
        json.dump({'note':'cycle_dice uses the same trained segmenter; not an independent accuracy measure. Test set is monitoring only, no automatic checkpoint selection.', 'cases':report},f,indent=2)
    return report
