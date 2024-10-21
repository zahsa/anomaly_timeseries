
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score
from sklearn.neighbors import KernelDensity
import argparse
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import cv2
import sys  
import time

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
sys.path.insert(0, './preprocessing')
sys.path.insert(0, './analysis')
import sal_anal
import dataset_prep_limit_vessel_type
import importlib
import preprocessing.dataset_prep_limit_vessel_type



importlib.reload(dataset_prep_limit_vessel_type)

importlib.reload(sal_anal)
from dataset_prep_limit_vessel_type import create_data

def main(args):

    v_type = args.v_type
    ds_name = args.ds_name
    attribute = args.AIS_attr
    traj_length = args.traj_length
    seq_length = args.seq_length
    contamination = args.contamination

    data = create_data(attribute = attribute, v_type = v_type, data_mode = 'all' , traj_length = traj_length , seq_length = seq_length, max_traj_num = None)

    data.read_data(ds_name = ds_name)

    df = data.dataset
    print('trajectory max', df['trajectory'].max())

    data.create_train_sc_traj()

   
    X = data.xtrain
    X = X.T
    traj_length = 10000
    window_size = 1000 # The window length
    n_col = traj_length - window_size + 1 # The number of columns in the trajectory matrix.

    total_salo_tp = []; total_salz_tp = []; total_ifor_tp = []; total_lof_tp = []
    total_salo_fp = []; total_salz_fp = []; total_ifor_fp = []; total_lof_fp = []

    total_salo_tn = []; total_salz_tn = []; total_ifor_tn = []; total_lof_tn = []
    total_salo_fn = []; total_salz_fn = []; total_ifor_fn = []; total_lof_fn = []
    
    ostu_time = []; zero_time = []; ifor_time = []; lof_time = []
    n_traj = X.shape[0]
    for imnum in range(0,n_traj): 
        print('traj num' , imnum)
        X0 = X[imnum]
#         print('trajectory shape',X0.shape)
        df_traj = pd.DataFrame(X0 , columns = [data.attribute])

        #==== isoloation forest model======
#         print('isolation forest')
        start_t = time.time()
        model =  IsolationForest(contamination=contamination)


        model.fit((df_traj[data.attribute]).to_numpy().reshape(-1,1))
        end_t = time.time()
        tot_time = end_t - start_t
        ifor_time.append(tot_time)
#         print('elapsed time:', tot_time)
        df_traj.reset_index(drop=True, inplace=True)



        df_traj['iforest_outliers']=pd.Series(model.predict(df_traj[data.attribute].to_numpy().reshape(-1,1))).apply(lambda x: 'Yes' if (x == -1) else 'No' )

        #==== saliency method =====
        # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns.
#         print('saliency -- ostu')
        start_t = time.time()
        Ximg = np.column_stack([X0[i:i+window_size] for i in range(0,n_col)])
        seg_inds, seg_vals , threshMap1 = sal_anal.get_outliers(Ximg,X0,window_size,'ostu')
        end_t = time.time()
        tot_time = end_t - start_t
        ostu_time.append(tot_time)
#         print('elapsed time:', tot_time)
        df_traj['sal_ostu_outliers']=pd.Series(range(0,len(X0))).apply(lambda x: 'Yes' if (x in seg_inds) else 'No' )

#         print('saliency -- zero')
        start_t = time.time()
        Ximg = np.column_stack([X0[i:i+window_size] for i in range(0,n_col)])
        seg_inds, seg_vals , threshMap1 = sal_anal.get_outliers(Ximg,X0,window_size,'zero_float',contamination=contamination)
        end_t = time.time()
        tot_time = end_t - start_t
        zero_time.append(tot_time)
#         print('elapsed time:',tot_time)
        
        df_traj['sal_zero_outliers']=pd.Series(range(0,len(X0))).apply(lambda x: 'Yes' if (x in seg_inds) else 'No' )

        
        bandwidth = 50
        anomality = sal_anal.anom_level(X0,bandwidth,kernel_name = 'gaussian')
        df_traj['anomality_level'] = pd.Series(anomality)



        #==== local outlier factor model======
#         print('local outlier factor')
        start_t = time.time()
        model = LocalOutlierFactor(contamination=contamination)
        model.fit_predict((df_traj[data.attribute]).to_numpy().reshape(-1,1))
        end_t = time.time()
        tot_time = end_t - start_t
        lof_time.append(tot_time)
#         print('elapsed time:',tot_time)
        
        df_traj.reset_index(drop=True, inplace=True)

        df_traj['lof_outliers']=pd.Series(model.fit_predict(df_traj[data.attribute].to_numpy().reshape(-1,1))).apply(lambda x: 'Yes' if (x == -1) else 'No' )

        df_traj.to_csv('/data/time_data_' + data.v_type + '_' + data.attribute + '_' + str(imnum) + '_contamination_' + str(10*contamination) + '.csv', index=False)
        
        
        # ==== evaluation based on the sum of anomality likelihood ======

        salo_tp,salo_tn,salo_fp,salo_fn = sal_anal.eval_outl(X0,df_traj,'sal_ostu_outliers')
           

        salz_tp,salz_tn,salz_fp,salz_fn = sal_anal.eval_outl(X0, df_traj,'sal_zero_outliers')
       

        ifor_tp,ifor_tn,ifor_fp,ifor_fn = sal_anal.eval_outl(X0, df_traj,'iforest_outliers')

  
        lof_tp,lof_tn,lof_fp,lof_fn = sal_anal.eval_outl(X0, df_traj,'lof_outliers')

        total_salo_tp.append(np.sum(salo_tp));total_salo_fp.append(np.sum(salo_fp))
        total_salz_tp.append(np.sum(salz_tp));total_salz_fp.append(np.sum(salz_fp))
        total_ifor_tp.append(np.sum(ifor_tp));total_ifor_fp.append(np.sum(ifor_fp))
        total_lof_tp.append(np.sum(lof_tp));total_lof_fp.append(np.sum(lof_fp))

        total_salo_tn.append(np.sum(salo_tn));total_salo_fn.append(np.sum(salo_fn))
        total_salz_tn.append(np.sum(salz_tn));total_salz_fn.append(np.sum(salz_fn))
        total_ifor_tn.append(np.sum(ifor_tn));total_ifor_fn.append(np.sum(ifor_fn))
        total_lof_tn.append(np.sum(lof_tn));total_lof_fn.append(np.sum(lof_fn))

    print('ostu_zero_sal')    
    print(data.v_type)
    print('salo total anom tp', np.sum(total_salo_tp))
    print('salz total anom tp', np.sum(total_salz_tp))

    print('ifor total anom tp', np.sum(total_ifor_tp))
    print('lof total anom tp', np.sum(total_lof_tp))
    print('================================')
    
    print('salo total anom tn', np.sum(total_salo_tn))
    print('salz total anom tn', np.sum(total_salz_tn))

    print('ifor total anom tn', np.sum(total_ifor_tn))
    print('lof total anom tn', np.sum(total_lof_tn))
    print('================================')
    
    print('salo total anom fp', np.sum(total_salo_fp))
    print('salz total anom fp', np.sum(total_salz_fp))

    print('ifor total anom fp', np.sum(total_ifor_fp))
    print('lof total anom fp', np.sum(total_lof_fp))
    print('================================')
    
    print('salo total anom fn', np.sum(total_salo_fn))
    print('salz total anom fn', np.sum(total_salz_fn))

    print('ifor total anom fn', np.sum(total_ifor_fn))
    print('lof total anom fn', np.sum(total_lof_fn))
    print('================================')
    
    print('ostu time', np.mean(ostu_time))
    print('zero_time', np.mean(zero_time))
    print('ifor_time', np.mean(ifor_time))
    print('lof_time', np.mean(lof_time))
    
# Accuracy = (TP+TN)/(TP+FP+FN+TN)
# Precision = TP/(TP+FP)
# Recall = TP/(TP+FN)
   
    dict_eval = {}
    dict_eval['TP_salo'] = np.sum(total_salo_tp)
    dict_eval['TP_salz'] = np.sum(total_salz_tp)
    dict_eval['TP_ifor'] = np.sum(total_ifor_tp)
    dict_eval['TP_lof'] = np.sum(total_lof_tp)

    dict_eval['FP_salo'] = np.sum(total_salo_fp)
    dict_eval['FP_salz'] = np.sum(total_salz_fp)
    dict_eval['FP_ifor'] = np.sum(total_ifor_fp)
    dict_eval['FP_lof'] = np.sum(total_lof_fp)
  
    dict_eval['TN_salo'] = np.sum(total_salo_tn)
    dict_eval['TN_salz'] = np.sum(total_salz_tn)
    dict_eval['TN_ifor'] = np.sum(total_ifor_tn)
    dict_eval['TN_lof'] = np.sum(total_lof_tn)

    dict_eval['FN_salo'] = np.sum(total_salo_fn)
    dict_eval['FN_salz'] = np.sum(total_salz_fn)
    dict_eval['FN_ifor'] = np.sum(total_ifor_fn)
    dict_eval['FN_lof'] = np.sum(total_lof_fn)
    
    dict_eval['tm_salo'] = np.mean(ostu_time)
    dict_eval['tm_salz'] = np.mean(zero_time)
    dict_eval['tm_ifor'] = np.mean(ifor_time)
    dict_eval['tm_lof'] = np.mean(lof_time)


    with open('/data/time_data_eval' + data.v_type + '_' + data.attribute + '_contamination_' + str(10*contamination) + '.pickle', 'wb') as handle:
        pickle.dump(dict_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)

        

if __name__ == '__main__':
    print('main')
    # Create the parser
    parser = argparse.ArgumentParser(description='sal based OD')
    # Add the arguments
    parser.add_argument('--v_type',
                        default='tanker',
                        type=str,
                        help='name of vessel type')
    
    parser.add_argument('--ds_name',
                        default='9class',
                        type=str,
                        help='name of dataset')

    parser.add_argument('--AIS_attr',
                        default='sog',
                        type=str,
                        help='AIS attribute')

    parser.add_argument('--traj_length',
                        default=10001,
                        type = int,
                        help='length of each trajectory')

    parser.add_argument('--seq_length',
                        default=300,
                        type=int,
                        help='length of each sequence')
    parser.add_argument('--contamination',
                        default = 0.1,
                        type = float,
                        help = 'threshold')
                        

    args = parser.parse_args()
    main(args)

