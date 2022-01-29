import torch
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.data.synthia.mono_dataset import SynthiaMonoDataset

cfg = cfg_from_file('config/Road_synthia_train_full.py')
d1 = SynthiaMonoDataset(cfg)
import pdb
pdb.set_trace()

dl = torch.utils.data.DataLoader(d1, collate_fn=d1.collate_fn, num_workers=8, batch_size=8, shuffle=True)

print(next(iter(d1)))
