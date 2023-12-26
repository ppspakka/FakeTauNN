import numpy as np
import torch


def nan_resampling(sample, gen, model, device):
    """a function to resample nan values in the generated samples

    Args:
        sample (np array): the generated samples
        gen (np array): the gen starting points
        model (torch nn): the flow model (must be on same device as below)
        device (string): the device to use for generating new samples

    Returns:
        sample: the full array with resampled nan values
    """    
    sample = torch.tensor(sample).to(device)
    gen = torch.tensor(gen).to(device)
    nan_mask = torch.isnan(sample).any(axis=1)
    if nan_mask.any():
        nan_idx = torch.nonzero(nan_mask)
        print(f"Resampling {len(nan_idx)} nans")
        # Generate new samples
        model.eval()
        while True:
            with torch.no_grad():
                sample[nan_idx] = model.sample(num_samples=1, context=gen[nan_mask])
                if not torch.isnan(sample[nan_idx]).any():
                    break
    sample = sample.detach().cpu().numpy()
    gen = gen.detach().cpu().numpy()
    return sample
