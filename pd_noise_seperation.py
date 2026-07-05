# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from utils.utils import plot_one_prpd,_pd_auto_cluster,read_data

def main():
    prpd_path, waveform_path = data_path+'/prpd.csv', data_path+'/waveform.csv'
    prpd_data, waveform_data = read_data(prpd_path, waveform_path)
    cluster_idx, cluster_num = _pd_auto_cluster(waveform_data, 16, 3)

    # PD-noise are seperated by different clusters
    prpd_cluster_0, waveform_cluster_0 = prpd_data[cluster_idx==0,:], waveform_data[cluster_idx==0,:]
    prpd_cluster_1, waveform_cluster_1 = prpd_data[cluster_idx==1,:], waveform_data[cluster_idx==1,:]

    print(cluster_num)
    plot_one_prpd(prpd_cluster_0, waveform_cluster_0)
    plt.show()
    plot_one_prpd(prpd_cluster_1, waveform_cluster_1)
    plt.show()

if __name__ == '__main__':
    data_path = './data/dataset-3'
    main()
