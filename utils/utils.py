# -*- coding: utf-8 -*-
import os
import math
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from kneed import KneeLocator
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(prpd_name, waveform_name):
    assert os.path.exists(prpd_name)
    assert os.path.exists(waveform_name)
    try:
        prpd_buf = np.loadtxt(prpd_name, delimiter='\t')
        waveform_buf = np.loadtxt(waveform_name, delimiter='\t')
        
        waveform_list = waveform_buf
        prpd_list = prpd_buf if prpd_buf.shape[0]==waveform_list.shape[0] else prpd_buf.T
        
        assert prpd_list.shape[0]==waveform_list.shape[0], 'prpd len and waveform len not match!'
        assert prpd_list.shape[1]==2, 'prpd shape[1] is not 2!'
    except Exception as e:
        return [], []
    
    return prpd_list, waveform_list

def _adp_sgfft(fea_fft, split_num):
    waveform_num,feadim_fft = fea_fft.shape
    frequency = np.array(range(0,feadim_fft)).reshape(1,feadim_fft)
    fea_centroid_list = (fea_fft*frequency).sum(axis=1)/(fea_fft[:,1:].sum(axis=1)+1e-10)
    
    #select frequency centroid minimum 1% as the finest centroid
    select_num = int(np.ceil(waveform_num*0.005))
    fea_centroid = int(fea_centroid_list[np.argsort(fea_centroid_list)[:select_num]].mean())
    
    #get the split
    split_num_left = min(fea_centroid, split_num//2)
    split_num_right = min(split_num-split_num_left,feadim_fft-fea_centroid)
    
    #get the feature
    fea_sgfft = []
    for idx in range(split_num_left+split_num_right):
        if idx < split_num_left:
            start_idx = int(idx*fea_centroid/split_num_left)
            end_idx = min(fea_centroid, int((idx+1)*fea_centroid/split_num_left))
        else:
            start_idx = fea_centroid+int((idx-split_num_left)*(feadim_fft-fea_centroid)/split_num_right)
            end_idx = min(feadim_fft, fea_centroid+int((idx+1-split_num_left)*(feadim_fft-fea_centroid)/split_num_right))
            
        if start_idx >= feadim_fft:
            break

        fea_tmp = fea_fft[:,start_idx:end_idx].mean(axis=1,keepdims=True)
        if idx==0:
            fea_sgfft = fea_tmp
        else:
            fea_sgfft = np.concatenate((fea_sgfft,fea_tmp), axis=1)
            
    return fea_sgfft

def extract_fea(waveform_data, fealen):
    waveform_data = np.array(waveform_data, dtype=np.float32)
    feadim = waveform_data.shape[1]
    
    #check if exists inf or nan
    if np.sum(np.isinf(waveform_data)) or np.sum(np.isnan(waveform_data)):
        raise Exception('waveform data exists nan or inf!')
    
    #get the fft feature
    fea_fft = np.abs(np.fft.fft(waveform_data))
    fea_fft[:,0] /= feadim
    fea_fft[:,1:] /= (feadim/2)
    fea_fft = np.array(fea_fft[:, 0:feadim//2+1], dtype=np.float32)

    #get the segmented fft
    fea_sgfft = _adp_sgfft(fea_fft, fealen)
    fea_sgfft = np.array(fea_sgfft, dtype=np.float32)
    
    return fea_sgfft

def _pd_auto_cluster(waveform_list, fea_len=32, test_cluster_num=8):
    
    #extract feature from the waveforms
    waveform_fea = extract_fea(waveform_list,fea_len)
    
    #call the kmeans
    scores = list()
    clusteridx_list = list()
    cluster_num_list = list()
    
    for i in range(2, test_cluster_num):
        kmeans_model = KMeans(n_clusters=i,n_init='auto').fit(waveform_fea)
        clusteridx = kmeans_model.predict(waveform_fea)
        labels = kmeans_model.labels_
        np.seterr(invalid='ignore')
        one_sse_score = kmeans_model.inertia_
        one_silhouette_score = silhouette_score(waveform_fea, labels) # [-1,1], higher better
        
        scores.append(one_silhouette_score) # one_sse_score
        clusteridx_list.append(clusteridx)
        cluster_num_list.append(i)
        
    best_idx = np.array(scores).argmax()
    clusteridx = clusteridx_list[best_idx]
    cluster_num = cluster_num_list[best_idx]
    
    return clusteridx, cluster_num

def plot_one_prpd(prpd_list, waveform_list=None, max_ampl=0, fig_title=''):        
    fontsize = 4
    fig = plt.figure(figsize=(5, 1.5), dpi=200)
    plt.rcParams['font.size']=fontsize  #font size

    ##########################plot one PRPD, sin and base line#########################
    ax = plt.subplot(1,2,1)
    max_ampl = max(np.abs(prpd_list[:,0])) if max_ampl==0 else max_ampl
    sin_x = np.arange(0,360+1,1)
    sin_y = np.arange(0,360+1,1.0)
    zero_y = np.zeros(361)
    for i in range(len(sin_x)):
        sin_y[i] = math.sin(2*math.pi/360*sin_x[i])*max_ampl
    plt.plot(sin_x, zero_y, color='grey',linewidth=0.2)
    plt.plot(sin_x, sin_y, color='grey',linewidth=0.2)

    #plot the colored PRPD
    try:
        xy = np.vstack([prpd_list[:,1],prpd_list[:,0]]).astype(np.float64)
        xy += np.random.random(xy.shape)*10e-10 #in case of singular matrix
        z = gaussian_kde(xy)(xy)
    except:
        return False
    sc = plt.scatter(prpd_list[:,1],prpd_list[:,0],c=z,s=0.1,marker=',',cmap=cm.jet)
    plt.xlim(left=0, right=360)
    plt.ylim(bottom=-max_ampl*1.2, top=max_ampl*1.2)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.xlabel('Phase'+r'$(^{\circ})$',fontsize=fontsize)
    plt.ylabel('Peak(V)',fontsize=fontsize)
    plt.title('PRPD-num-'+str(len(prpd_list)), fontsize=fontsize+1, color='blue')

    ##########################plot the typical waveforms#########################
    ax = plt.subplot(1,2,2)
    if waveform_list is not None:
        random_idx = np.random.permutation(len(waveform_list))[0]
        plt.plot(waveform_list[random_idx],linewidth=0.3)
        plt.xlim(left=0, right=waveform_list.shape[1])
        plt.ylim(bottom=-max_ampl*1.2, top=max_ampl*1.2)
        plt.xlabel('Time',fontsize=fontsize)
        plt.ylabel('Amplitude(V)',fontsize=fontsize)
        plt.title('Typical pulse '+str(random_idx), fontsize=fontsize+1, color='blue')

    ############adjust the distance###########
    plt.tight_layout()
    plt.suptitle(fig_title,fontsize=fontsize+1,y=1.05)

    return True

def _pairwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances

def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices
