import os
import pickle
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
import plotly.express as px
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import cv2



def prob_density(traj,kernel_name = 'gaussian'):
    x = traj.reshape(-1, 1)


    kde = KernelDensity(bandwidth=np.max(x)/50, kernel= kernel_name )
    kde.fit(x)

    logprob = kde.score_samples(x)

    kde_pdf = np.exp(logprob)


    return kde_pdf


def standardization(x_input):
    training_min = x_input.min()
    training_max = x_input.max()
    x_standardized = (x_input - training_min) / (training_max - training_min)
#     print("Number of training samples:", len(x_standardized))
    return x_standardized

def anom_level(traj,bw = 100, kernel_name = 'gaussian'):

    x = traj.reshape(-1, 1)
    # we sort it just for the sake of plotting

    kde = KernelDensity(bandwidth=np.max(x)/bw, kernel= kernel_name )
    kde.fit(x)

    logprob = kde.score_samples(x)

    pdf = np.exp(logprob)

    pdf_norm=standardization(pdf)
    
    return 1-pdf_norm


def find_outlier_inds(x,n_seg,outlier_thresh):
    saliencymap_scaled = x
    smc = saliencymap_scaled.T

    sal_ind = lambda x:np.where(x>= outlier_thresh)

    anom_ind = [[] for _ in range(n_seg)]
    for i in range(0,n_seg): 
        anom_ind[i].append(sal_ind(smc[i])[0].tolist())
  
    anom_ind_correct = [[] for _ in range(n_seg)]

    for i in range(0,n_seg):
        a = [x + i for x in anom_ind[i][0]]
        anom_ind_correct[i].append(a)
    
    # convert anom_ind_correct datatype to 2D array
    al = []
    for a in anom_ind_correct:
  
        al = np.append(al, np.asarray(a))

    # flatten the 2D array into 1D array
    alf = al.flatten()
    
    # get the unique anomaly indices
    ualf = np.unique(alf)
#     print('number of all unique anom indices: ', ualf.shape)    
    return(ualf)


def create_salmap(Ximg,tmax):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(Ximg)
    threshMap = cv2.threshold((saliencyMap * tmax).astype("uint8"), 0, tmax, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    min_max_scaler = preprocessing.MinMaxScaler()
    threshmap_scaled = min_max_scaler.fit_transform(threshMap)
    saliencymap_scaled = min_max_scaler.fit_transform(saliencyMap)
    Ximg_scaled = min_max_scaler.fit_transform(Ximg)

    return(saliencymap_scaled,threshmap_scaled,Ximg_scaled)



def get_outliers(Ximg,X0,window_size,method, **kwargs):
    if 'contamination' in kwargs.keys():
            contamination = kwargs['contamination']
            
    if method == 'zero_float':
        output_map = cv2.threshold((Ximg*1).astype("float64"), 0, 1, cv2.THRESH_TOZERO )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row >= contamination)
        segs2 = np.argwhere(last_row >= contamination)  
    
    if method == 'zero':
        output_map = cv2.threshold((Ximg*255).astype("uint8"), 0, 255, cv2.THRESH_TOZERO )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row >= contamination)#100
        segs2 = np.argwhere(last_row >= contamination) #100 
    
    if method == 'trunc_float':
        output_map = cv2.threshold((Ximg*1).astype("float"), 0, 1, cv2.THRESH_TRUNC )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row >= contamination)
        segs2 = np.argwhere(last_row >= contamination) 
        
    if method == 'trunc':
        output_map = cv2.threshold((Ximg*255).astype("uint8"), 0, 255, cv2.THRESH_TRUNC )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row >= 10)
        segs2 = np.argwhere(last_row >= 10)  
        
    if method == 'adaptive_float':
        output_map = cv2.threshold((Ximg*1).astype("float64"), 0, 1, cv2.ADAPTIVE_THRESH_MEAN_C )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 1) 
        segs2 = np.argwhere(last_row == 1)
        
    if method == 'adaptive_float':
        output_map = cv2.threshold((Ximg*1).astype("float64"), 0, 1, cv2.ADAPTIVE_THRESH_MEAN_C )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 1) 
        segs2 = np.argwhere(last_row == 1)  
        
    if method == 'adaptive':
        output_map = cv2.threshold((Ximg*255).astype("uint8"), 0, 255, cv2.ADAPTIVE_THRESH_MEAN_C )[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 255) 
        segs2 = np.argwhere(last_row == 255)   
    
    if method == 'ostu_float':   
        print('ostu')

        output_map = cv2.threshold((Ximg*1).astype("float64"), 0, 1, cv2.THRESH_OTSU)[1]

        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 1)
        segs2 = np.argwhere(last_row == 1) 
        
    if method == 'ostu':   
        print('ostu')

        output_map = cv2.threshold((Ximg*255).astype("uint8"), 0, 255, cv2.THRESH_OTSU)[1]



        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 255)
        segs2 = np.argwhere(last_row == 255)  

    if method == 'adaptive_threshmap':
        print('adaptive')
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(Ximg)
        output_map = cv2.adaptiveThreshold((saliencyMap * 255).astype("uint8"), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row == 255)
        segs2 = np.argwhere(last_row == 255)  

    if method == 'saliecny':
        print('saliency')
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, output_map) = saliency.computeSaliency(Ximg)
        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row > 0.3)
        segs2 = np.argwhere(last_row > 0.3)
    
    if method =='scaled_saliecny':
        print('scaled saliency')
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(Ximg)
        min_max_scaler = preprocessing.MinMaxScaler()
        output_map = min_max_scaler.fit_transform(saliencyMap)
        first_row = output_map[0]
        last_row = output_map[-1]
        segs1 = np.argwhere(first_row > 0.99)
        segs2 = np.argwhere(last_row > 0.99)

    segs2 = segs2 + window_size - 1
    seg_inds = np.concatenate((segs1,segs2))
    seg_inds = np.unique(seg_inds)
    seg_vals = []
    
    for i in seg_inds:
        seg_vals.append(X0[int(i)])
    return seg_inds,seg_vals,output_map

def eval_outl(X0, df_traj,method_name):
    anomality = df_traj['anomality_level'] 
    tp = 0 
    fp = 0 
    tn = 0; 
    fn = 0
    v_outl=[]; i_outl=[]
    for i,v in enumerate(df_traj[method_name]):
        if v == 'Yes': # for the detected points (anomaly points)
            v_outl.append(X0[i])
            i_outl.append(i)
            tp += anomality[i]   
            fp += anomality[i]-1 
        else: 
            tn += anomality[i]-1 
            fn += anomality[i]
 
    return(tp,tn,fp,fn)
