
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def tagROC(reco, gen_partonFlavour_b, gen_partonFlavour_uds,  tag_name):
  """ Compute the ROC curve for a given b-tagging on gen parton flavour
    inputs:
    reco: reconstructed jets DF
    gen_partonFlavour: gen parton flavour 
    tag_name: name of the b-tagging column to be used

    returns:
    fpr: false positive rate
    tpr: true positive rate
    roc_auc: area under the curve
    bs: b-tagged jets
    nbs: non-b-tagged jets"""
  
  mask_b = np.array(gen_partonFlavour_b.flatten(), dtype=bool)
  mask_uds = np.array(gen_partonFlavour_uds.flatten(), dtype=bool)
  bs = reco[tag_name].values[mask_b].flatten()
  nbs = reco[tag_name].values[mask_uds].flatten()
  # nbs = nbs[0:len(bs)]

  bs = bs[bs >=-0.05]
  nbs = nbs[nbs >=-0.05]

  bs = np.where(bs<0, 0, bs)
  nbs = np.where(nbs<0, 0, nbs)

  bs = np.where(bs>1, 1, bs)
  nbs = np.where(nbs>1, 1, nbs)

  y_bs = np.ones(len(bs))
  y_nbs = np.zeros(len(nbs))
  y_t = np.concatenate((y_bs, y_nbs))
  y_s = np.concatenate((bs, nbs))

  fpr, tpr, _ = roc_curve(y_t.ravel(), y_s.ravel())
  roc_auc = auc(fpr, tpr)

  return fpr, tpr, roc_auc, bs, nbs

def profile_hist(n_bins, x_arr, y_arr):
  """ Compute the profile histogram of a 2D distribution
  inputs:
  n_bins: number of bins
  x_arr: x values
  y_arr: y values
  
  returns:
  x_slice_mean: mean of each vertical slice of the 2D distribution
  x_slice_rms: RMS of each vertical slice of the 2D distribution
  xbinwn: bin width of the x axis
  xe: bin edges of the x axis
  """
  n = n_bins
  y = y_arr.flatten()
  x = x_arr.flatten()
  x_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), n)
  y_bins = np.logspace(np.log10(y.min()), np.log10(y.max()), n)
  H, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])

  xbinwn = xe[1]-xe[0]

  # getting the mean and RMS values of each vertical slice of the 2D distribution
  x_slice_mean, x_slice_rms = [], []
  for i,b in enumerate(xe[:-1]):
      x_slice_mean.append( y[ (x>xe[i]) & (x<=xe[i+1]) ].mean())
      x_slice_rms.append( y[ (x>xe[i]) & (x<=xe[i+1]) ].std())
      
  x_slice_mean = np.array(x_slice_mean)
  x_slice_rms = np.array(x_slice_rms)

  return x_slice_mean, x_slice_rms, xbinwn, xe