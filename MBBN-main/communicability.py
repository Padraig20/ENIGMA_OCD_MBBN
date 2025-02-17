import nitime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit
import scipy.stats as stats
import pywt
import networkx as nx
import argparse
from sktime.libs.vmdpy import VMD
from time_series_utils import *

def get_arguments(base_path = os.getcwd()):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROI_num', type=int,default=400, choices=[180, 304, 316, 400])
    # parser.add_argument('--ukb_path', default='/scratch/connectome/stellasybae/UKB_ROI') ## labserver
    parser.add_argument('--ukb_path', default='/pscratch/sd/p/pakmasha/UKB_304_ROIs') ## Perlmutter
    parser.add_argument('--abcd_path', default='/storage/bigdata/ABCD/fmriprep/1.rs_fmri/5.ROI_DATA') ## labserver
    parser.add_argument('--enigma_path', default='/pscratch/sd/p/pakmasha/MBBN_data') ## Perlmutter
    parser.add_argument('--dataset_name', type=str, choices=['ABCD', 'UKB', 'ENIGMA_OCD'], default="ABCD")
    parser.add_argument('--base_path', type=str, default=os.getcwd())
    args = parser.parse_args()
        
    return args

def wavelet_corr_mat(signal):
    # signal shape :  (ROI_num, seq_len)

    # wavelet transformation
    coeffs = pywt.dwt(signal, 'db1')  # 'db1' =  Daubechies wavelet
    cA, cD = coeffs  # cA: Approximation Coefficients, cD: etail Coefficients

    return np.corrcoef(cA)  # compute correlation matrix using approximation coefficients

def create_network(correlation_matrix, threshold=0.2):
    # Generate graph whose size is equivalent to correlation matrix
    G = nx.Graph()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            # add edge when correlation coefficient > threshold.
            if np.abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)
    return G

base_path = os.getcwd() 
args = get_arguments(base_path)
print(f"base_path: {args.base_path}")

if args.dataset_name == 'ABCD':
    data_dir = args.abcd_path
    TR = 0.8
    seq_len = 348
    subject = open(f'{args.base_path}/splits/ABCD/ABCD_reconstruction_ROI_{args.ROI_num}_seq_len_{seq_len}_split1.txt', 'r').readlines()
    subject = [x[:-1] for x in subject]
    subject.remove('train_subjects')
    subject.remove('val_subjects')
    subject.remove('test_subjects')


elif args.dataset_name == 'UKB':
    data_dir = args.ukb_path
    TR = 0.735
    seq_len = 464
    # subject = open(f'{args.base_path}/splits/UKB/UKB_reconstruction_ROI_{args.ROI_num}_seq_len_{seq_len}_split1.txt', 'r').readlines()
    subject = open("/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/splits/UKB/UKB_reconstruction_ROI_400_seq_len_464_split1.txt", 'r').readlines()
    subject = [x[:-1] for x in subject]
    subject.remove('train_subjects')
    subject.remove('val_subjects')
    subject.remove('test_subjects')

elif args.dataset_name == 'ENIGMA_OCD':
    data_dir = args.enigma_path
    seq_len = 700
    subject = open(f'{args.base_path}/splits/ENIGMA_OCD/ENIGMA_OCD_reconstruction_ROI_{args.ROI_num}_seq_len_{seq_len}_split1.txt', 'r').readlines()
    subject = [x[:-1] for x in subject]
    subject.remove('train_subjects')
    subject.remove('val_subjects')
    subject.remove('test_subjects')

if args.ROI_num == 400:
    ROI_name = 'Schaefer400'
elif args.ROI_num == 180:
    ROI_name = 'HCPMMP1'
elif args.ROI_num == 316:
    ROI_name = '316'
elif args.ROI_num == 304:
    ROI_name = '304'
    
subject = subject[:-1]
print('number of subject', len(subject))
num_processes = cpu_count()
print('number of processes', num_processes)

n = args.ROI_num
imf1_comm_mat_whole = np.zeros((n, n))
imf2_comm_mat_whole = np.zeros((n, n))
imf3_comm_mat_whole = np.zeros((n, n))
imf4_comm_mat_whole = np.zeros((n, n))


def main(sub):
    try:
        path_to_fMRIs = os.path.join(data_dir, sub, 'schaefer_400Parcels_17Networks_'+sub+'.npy')
        y = np.load(path_to_fMRIs)[20:20+seq_len].T

        sample_whole = np.zeros((seq_len))
        for i in range(n):
            sample_whole+=y[i]

        sample_whole /= n    

        T = TimeSeries(sample_whole, sampling_interval=TR)
        S_original = SpectralAnalyzer(T)

        # Lorentzian function fitting
        xdata = np.array(S_original.spectrum_fourier[0][1:])
        ydata = np.abs(S_original.spectrum_fourier[1][1:])

        def lorentzian_function(x, s0, corner):
            return (s0*corner**2) / (x**2 + corner**2)

        p0 = [0, 0.006]

        popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000)

        f1 = popt[1]

        knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))


        def modified_lorentzian_function(x, beta_low, beta_high, A, B, corner):
            return np.where(x < corner, A * x**beta_low, B * x**beta_high)
            #return A*x**(-beta_low) / (1+(x/corner)**beta_high)

        p1 = [2, 1, 23, 25, 0.16]

        popt_mo, pcov = curve_fit(modified_lorentzian_function, xdata[knee:], ydata[knee:], p0=p1, maxfev = 50000)
        pink = round(popt_mo[-1]/(1/(sample_whole.shape[0]*TR)))
        f2 = popt_mo[-1]

        # 01 high ~ (low+ultralow)
        T1 = TimeSeries(y, sampling_interval=TR)
        S_original1 = SpectralAnalyzer(T1)
        FA1 = FilterAnalyzer(T1, lb= f2)
        high = stats.zscore(FA1.filtered_boxcar.data, axis=1)
        ultralow_low = FA1.data-FA1.filtered_boxcar.data

        # 02 low ~ ultralow
        T2 = TimeSeries(ultralow_low, sampling_interval=TR)
        S_original2 = SpectralAnalyzer(T2)
        FA2 = FilterAnalyzer(T2, lb=f1)
        low = stats.zscore(FA2.filtered_boxcar.data, axis=1)
        ultralow = stats.zscore(FA2.data-FA2.filtered_boxcar.data, axis=1)

        high_G = create_network(wavelet_corr_mat(high))
        high_comm = nx.communicability(high_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = high_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        high_comm_mat_whole=communicability_matrix

        low_G = create_network(wavelet_corr_mat(low))
        low_comm = nx.communicability(low_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = low_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        low_comm_mat_whole=communicability_matrix

        ultralow_G = create_network(wavelet_corr_mat(ultralow))
        ultralow_comm = nx.communicability(ultralow_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = ultralow_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        ultralow_comm_mat_whole=communicability_matrix
    except:
        high_comm_mat_whole=np.zeros((n, n))
        low_comm_mat_whole=np.zeros((n, n))
        ultralow_comm_mat_whole=np.zeros((n, n))
    
    return high_comm_mat_whole, low_comm_mat_whole, ultralow_comm_mat_whole

def main_enigma(sub):
    try:
        path_to_fMRIs = os.path.join(os.path.join(data_dir, sub, sub+'.npy'))
        y = np.load(path_to_fMRIs, mmap_mode="r")[20:].T

        if y.shape[1] > seq_len:
            y = y[:, :seq_len]

        ts_length = y.shape[1]
        site = sub.split('_')[-2]   
        TR = repetition_time(site)

        # average the time series across ROIs
        sample_whole = np.zeros(ts_length,)
        n = y.shape[0]

        for i in range(n):
            sample_whole+=y[i]

        sample_whole /= n 

        # VMD setting
        f = sample_whole

        # z-score normalization
        f = (f - np.mean(f)) / np.std(f)
        if len(f)%2:
            f = f[:-1]

        # VMD parameters
        K = 4             # number of modes
        DC = 0            # no DC part imposed
        init = 0          # initialize omegas uniformly
        tol = 1e-7        # convergence tolerance
        alpha = 100
        tau = 3.5
        u, _, omega = VMD(f, alpha, tau, K, DC, init, tol)

        band_cutoffs = compute_imf_bandwidths(u, omega, 1/TR)

        if band_cutoffs['imf1_lb'] > band_cutoffs['imf1_hb']:
            raise ValueError(f"band_cutoffs['imf1_lb'] {band_cutoffs['imf1_lb']} is larger than band_cutoffs['imf1_hb'] {band_cutoffs['imf1_hb']} for subject {subj_name}")
        elif band_cutoffs['imf1_lb'] == band_cutoffs['imf1_hb']:
            imf1 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf1 = bandpass_filter_2d(y, band_cutoffs['imf1_lb'], band_cutoffs['imf1_hb'], 1/TR)
            imf1 = stats.zscore(imf1, axis=1)

        if band_cutoffs['imf2_lb'] > band_cutoffs['imf2_hb']:
            raise ValueError(f"band_cutoffs['imf2_lb'] {band_cutoffs['imf2_lb']} is larger than band_cutoffs['imf2_hb'] {band_cutoffs['imf2_hb']} for subject {subj_name}")
        elif band_cutoffs['imf2_lb'] == band_cutoffs['imf2_hb']:
            imf2 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf2 = bandpass_filter_2d(y, band_cutoffs['imf2_lb'], band_cutoffs['imf2_hb'], 1/TR)
            imf2 = stats.zscore(imf2, axis=1)

        if band_cutoffs['imf3_lb'] > band_cutoffs['imf3_hb']:
            raise ValueError(f"band_cutoffs['imf3_lb'] {band_cutoffs['imf3_lb']} is larger than band_cutoffs['imf3_hb'] {band_cutoffs['imf3_hb']} for subject {subj_name}")
        elif band_cutoffs['imf3_lb'] == band_cutoffs['imf3_hb']:
            imf3 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf3 = bandpass_filter_2d(y, band_cutoffs['imf3_lb'], band_cutoffs['imf3_hb'], 1/TR)
            imf3 = stats.zscore(imf3, axis=1)

        if band_cutoffs['imf4_lb'] > band_cutoffs['imf4_hb']:
            raise ValueError(f"band_cutoffs['imf4_lb'] {band_cutoffs['imf4_lb']} is larger than band_cutoffs['imf4_hb'] {band_cutoffs['imf4_hb']} for subject {subj_name}")
        elif band_cutoffs['imf4_lb'] == band_cutoffs['imf4_hb']:
            imf4 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf4 = bandpass_filter_2d(y, band_cutoffs['imf4_lb'], band_cutoffs['imf4_hb'], 1/TR)
            imf4 = stats.zscore(imf4, axis=1)

        imf1_G = create_network(wavelet_corr_mat(imf1))
        imf1_comm = nx.communicability(imf1_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf1_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf1_comm_mat_whole=communicability_matrix

        imf2_G = create_network(wavelet_corr_mat(imf2))
        imf2_comm = nx.communicability(imf2_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf2_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf2_comm_mat_whole=communicability_matrix

        imf3_G = create_network(wavelet_corr_mat(imf3))
        imf3_comm = nx.communicability(imf3_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf3_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf3_comm_mat_whole=communicability_matrix

        imf4_G = create_network(wavelet_corr_mat(imf4))
        imf4_comm = nx.communicability(imf4_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf4_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf4_comm_mat_whole=communicability_matrix

    except:
        imf1_comm_mat_whole=np.zeros((n, n))
        imf2_comm_mat_whole=np.zeros((n, n))
        imf3_comm_mat_whole=np.zeros((n, n))
        imf4_comm_mat_whole=np.zeros((n, n))

    return imf1_comm_mat_whole, imf2_comm_mat_whole, imf3_comm_mat_whole, imf4_comm_mat_whole

def main_ukb(sub):
    try:
        path_to_fMRIs = os.path.join(os.path.join(data_dir, sub, 'schaefer_400Parcels_17Networks_'+sub+'.npy'))
        y = np.load(path_to_fMRIs, mmap_mode="r")[20:].T
        print(f"y.shape: {y.shape}")

        if y.shape[1] > seq_len:
            y = y[:, :seq_len]

        ts_length = y.shape[1]

        # average the time series across ROIs
        sample_whole = np.zeros(ts_length,)
        n = y.shape[0]

        for i in range(n):
            sample_whole+=y[i]

        sample_whole /= n 

        # VMD setting
        f = sample_whole

        # z-score normalization
        f = (f - np.mean(f)) / np.std(f)
        if len(f)%2:
            f = f[:-1]

        # VMD parameters
        K = 4             # number of modes
        DC = 0            # no DC part imposed
        init = 0          # initialize omegas uniformly
        tol = 1e-7        # convergence tolerance
        alpha = 100
        tau = 3.5
        u, _, omega = VMD(f, alpha, tau, K, DC, init, tol)

        band_cutoffs = compute_imf_bandwidths(u, omega, 1/TR)
        print(f"band_cutoffs: {band_cutoffs}")

        if band_cutoffs['imf1_lb'] > band_cutoffs['imf1_hb']:
            raise ValueError(f"band_cutoffs['imf1_lb'] {band_cutoffs['imf1_lb']} is larger than band_cutoffs['imf1_hb'] {band_cutoffs['imf1_hb']} for subject {subj_name}")
        elif band_cutoffs['imf1_lb'] == band_cutoffs['imf1_hb']:
            imf1 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf1 = bandpass_filter_2d(y, band_cutoffs['imf1_lb'], band_cutoffs['imf1_hb'], 1/TR)
            imf1 = stats.zscore(imf1, axis=1)

        if band_cutoffs['imf2_lb'] > band_cutoffs['imf2_hb']:
            raise ValueError(f"band_cutoffs['imf2_lb'] {band_cutoffs['imf2_lb']} is larger than band_cutoffs['imf2_hb'] {band_cutoffs['imf2_hb']} for subject {subj_name}")
        elif band_cutoffs['imf2_lb'] == band_cutoffs['imf2_hb']:
            imf2 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf2 = bandpass_filter_2d(y, band_cutoffs['imf2_lb'], band_cutoffs['imf2_hb'], 1/TR)
            imf2 = stats.zscore(imf2, axis=1)

        if band_cutoffs['imf3_lb'] > band_cutoffs['imf3_hb']:
            raise ValueError(f"band_cutoffs['imf3_lb'] {band_cutoffs['imf3_lb']} is larger than band_cutoffs['imf3_hb'] {band_cutoffs['imf3_hb']} for subject {subj_name}")
        elif band_cutoffs['imf3_lb'] == band_cutoffs['imf3_hb']:
            imf3 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf3 = bandpass_filter_2d(y, band_cutoffs['imf3_lb'], band_cutoffs['imf3_hb'], 1/TR)
            imf3 = stats.zscore(imf3, axis=1)

        if band_cutoffs['imf4_lb'] > band_cutoffs['imf4_hb']:
            raise ValueError(f"band_cutoffs['imf4_lb'] {band_cutoffs['imf4_lb']} is larger than band_cutoffs['imf4_hb'] {band_cutoffs['imf4_hb']} for subject {subj_name}")
        elif band_cutoffs['imf4_lb'] == band_cutoffs['imf4_hb']:
            imf4 = np.zeros((y.shape[0], y.shape[1]))
        else:
            imf4 = bandpass_filter_2d(y, band_cutoffs['imf4_lb'], band_cutoffs['imf4_hb'], 1/TR)
            imf4 = stats.zscore(imf4, axis=1)

        imf1_G = create_network(wavelet_corr_mat(imf1))
        imf1_comm = nx.communicability(imf1_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf1_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf1_comm_mat_whole=communicability_matrix

        imf2_G = create_network(wavelet_corr_mat(imf2))
        imf2_comm = nx.communicability(imf2_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf2_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf2_comm_mat_whole=communicability_matrix

        imf3_G = create_network(wavelet_corr_mat(imf3))
        imf3_comm = nx.communicability(imf3_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf3_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf3_comm_mat_whole=communicability_matrix

        imf4_G = create_network(wavelet_corr_mat(imf4))
        imf4_comm = nx.communicability(imf4_G)
        communicability_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    communicability_matrix[i][j] = imf4_comm[i][j]
                except:
                    communicability_matrix[i][j] = 0
        imf4_comm_mat_whole=communicability_matrix

    except:
        imf1_comm_mat_whole=np.zeros((n, n))
        imf2_comm_mat_whole=np.zeros((n, n))
        imf3_comm_mat_whole=np.zeros((n, n))
        imf4_comm_mat_whole=np.zeros((n, n))

    return imf1_comm_mat_whole, imf2_comm_mat_whole, imf3_comm_mat_whole, imf4_comm_mat_whole

if args.dataset_name == 'ENIGMA_OCD':
    pool = Pool(num_processes)
    results = pool.map(main_enigma, subject)
elif args.dataset_name == 'UKB':
    pool = Pool(num_processes)
    results = pool.map(main_ukb, subject)

sub_num = len(subject)

imf1_comm_mat_whole = sum([results[i][0] for i in range(sub_num)]) / len(subject)
imf2_comm_mat_whole = sum([results[i][1] for i in range(sub_num)]) / len(subject)
imf3_comm_mat_whole = sum([results[i][2] for i in range(sub_num)]) / len(subject)
imf4_comm_mat_whole = sum([results[i][3] for i in range(sub_num)]) / len(subject)

np.save(f'./data/communicability/{args.dataset_name}_imf1_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(imf1_comm_mat_whole, axis=1)))
np.save(f'./data/communicability/{args.dataset_name}_imf2_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(imf2_comm_mat_whole, axis=1)))
np.save(f'./data/communicability/{args.dataset_name}_imf3_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(imf3_comm_mat_whole, axis=1)))
np.save(f'./data/communicability/{args.dataset_name}_imf4_comm_ROI_order_{ROI_name}.npy', np.argsort(np.sum(imf4_comm_mat_whole, axis=1)))
# last ROI has highest communicability