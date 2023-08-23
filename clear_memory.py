import torch
import gc

torch.cuda.empty_cache()
gc.collect()
print("CUDA memory freed and garbage collected.")