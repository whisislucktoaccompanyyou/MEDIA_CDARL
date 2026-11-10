"""Prepare only fractal labels and a patient-grouped Kaggle manifest."""
import json, random
from pathlib import Path
import cv2
import numpy as np
from prepare_fundus_data import fractal_mask
root=Path('data/Dataset_Kaggle')
source=Path('../data/kaggle/extracted/train')
root.mkdir(parents=True,exist_ok=True)
names=sorted(p.name for p in source.glob('*.jpeg'))
assert len(names)==35126, len(names)
patients=sorted(set(n.split('_')[0] for n in names))
random.Random(2026).shuffle(patients)
a,b=int(len(patients)*.9),int(len(patients)*.95)
groups={p:('train' if i<a else 'val' if i<b else 'test') for i,p in enumerate(patients)}
manifest={split:[n for n in names if groups[n.split('_')[0]]==split] for split in ['train','val','test']}
(root/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
out=root/'train/trainA';out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(2026)
for i in range(1000):
    path=out/('fractal_%04d.png'%i)
    mask=fractal_mask(rng)
    if not path.exists():
        assert cv2.imwrite(str(path),mask)
print('Kaggle only splits:',{k:len(v) for k,v in manifest.items()},flush=True)
