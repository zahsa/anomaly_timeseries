import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



def difference_method(x_input):
    x_transformed = x_input.diff().dropna()
    return x_transformed


def standardization(x_input):
    training_min = x_input.min()
    training_max = x_input.max()
    x_standardized = (x_input - training_min) / (training_max - training_min)
#     print("Number of training samples:", len(x_standardized))
    return x_standardized

# Generated training sequences for use in the model. The length of each sequence is time_steps.
def create_sequences(values, time_steps):
    output = []
    print('len: ', len(values))
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
#         print(i)
    return np.stack(output)

class create_data:
    
    def __init__(self, attribute = 'sog', v_type = 'fishing' , data_mode = 'all' , traj_length = 10001 , seq_length = 300, max_traj_num = 100):
        
        self.attribute = attribute
        self.data_mode = data_mode
        self.traj_length = traj_length
        self.seq_length = seq_length
        self.max_traj_num = max_traj_num
        self.v_type = v_type
        
    def read_data(self,ds_name):
        print('-- read dataset --')
        
        self.ds_name = ds_name
        if ds_name == 'Tanker':
            dataset = pd.read_csv('/dataset/vessels_Tankers.csv', parse_dates=['time'])
        elif ds_name == '4class':
             dataset = pd.read_csv('/dataset/vessels_Classes[30, 37, 80, 60].csv', parse_dates=['time'])
        elif ds_name == '9class':
             dataset = pd.read_csv('/dataset/vessels_Classes[30, 32, 34, 36, 37, 52, 60, 71, 80].csv', parse_dates=['time'])

    
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        df = dataset.sort_values(by=['trajectory', "time"])
        df = df.dropna()
        if self.v_type == 'fishing':
            df = df.loc[dataset['vessel_type'] == 30]
            self.dataset = df
        if self.v_type == 'pleasure':
            df = df.loc[dataset['vessel_type'] == 37]
            self.dataset = df
        if self.v_type == 'cargo':
            df = df.loc[dataset['vessel_type'] == 60]
            self.dataset = df
        if self.v_type == 'tanker':
            df = df.loc[dataset['vessel_type'] == 80]
            self.dataset = df
        if self.v_type == 'tug':
            df = df.loc[dataset['vessel_type'] == 52]
            self.dataset = df

        if self.v_type == 'special':
            df = df.loc[dataset['vessel_type'] == 34]
            self.dataset = df
            
        if self.v_type == 'sailing':
            df = df.loc[dataset['vessel_type'] == 36]
            self.dataset = df
            
        if self.max_traj_num is not None:
            self.n_vessels = self.max_traj_num
        else:
            self.n_vessels = df['trajectory'].max()
#         n_traj = 2208
#         import ipdb;ipdb.set_trace()
        n_traj = 0
        ind_traj = []
        for ti in range(self.n_vessels):
            ds = df[df['trajectory'] == ti] # get the time series trajectory corresponding to one vessel
            ds_traj = ds.loc[:,self.attribute]
            ds_traj.dropna()
#             print('isnan', np.isnan(ds_traj))
            # limit the min and max length
#             print(ds_traj.shape[0])
            if ds_traj.shape[0] >= 1000: #and ds_traj.isnull().values.any(): 
                
                n_traj = n_traj + 1
                ind_traj.append(ti)
#                 print(f"vessel number {ti} selected, with shape {ds_traj.shape}")
      
        self.n_traj = n_traj
        self.traj_ind = ind_traj
        print(f'number of vessels selected is {self.n_traj}')
        print('indices of selected vessels', self.traj_ind)
        print(f'number of all vessels is {self.n_vessels}')
#         import ipdb;ipdb.set_trace()
        # This is problematic. We need to check the number of samples from each class and split each class to train and test sets
#         self.train_ind = list(range(round(self.n_traj * 0.8)))
#         self.test_ind = list(range(round(self.n_traj * 0.8) + 1,self.n_traj))

    def print_ds_info(self):
        print('number of vessels selected' , self.n_traj)
        print('indices of selected vessels', self.traj_ind)
        print('maximum trajectory length' , self.traj_length)
        print('sequence length', self.seq_length)
        
    def create_test_train_ind(self): 
        print('-- create train and test indices --')

        train_ind = list(range(round(self.n_traj * 0.8)))
        test_ind = list(range(round(self.n_traj * 0.8) + 1,self.n_traj))
        self.train_ind = [self.traj_ind[i] for i in train_ind]
        self.test_ind = [self.traj_ind[i] for i in test_ind]
        
#         if not os.path.exists(self.train_path):
            
    def create_traj_seqs(self):
        ''' create segmented train data with original trajectory values --- useful for AE '''
        ''' variable size trajectory '''
        ''' No standardization, no stationarization '''  
                
        print(f'-- create traj sequences for data mode {self.data_mode} --')

        if self.data_mode == 'train':
            vessel_inds = self.train_ind
        elif self.data_mode == 'all':
            vessel_inds = self.traj_ind
            
        train_num = 0
        for t,ti in enumerate(vessel_inds):
            print('vessel number ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,self.variable]
            df_sog = pd.DataFrame(df_sog)
            
            # limit trajectory length
            if df_sog > self.traj_length:
                df_sog = df_sog[0:self.traj_length]

            x_train_seq = create_sequences(df_sog.values, self.seq_length)

            if np.shape(np.argwhere(np.isnan(x_train_seq)))[0] > 0:
                print('nan values !! ---------------',t)
                continue
            if train_num==0:
                xx_train = x_train_seq

            if train_num > 0:
                xx_train = np.concatenate((xx_train,x_train_seq),axis=0)
            train_num +=1

#             print('num of training samples', train_num)
        
      
        print('num of training vessels taken for seqs', train_num)
        print('num of all generated sequences', xx_train.shape)

        self.xtrain = xx_train
        self.train_num = train_num
        
    
    

        
    def create_sc_traj_seqs(self):
            ''' create train data sequences  --- good for AE''' 
            ''' variable size trajectory '''
            ''' not considering time steps beyond traj_length point '''
            ''' trajectories are scaled between 0 and 1 '''
            
             
            print(f'-- create scaled traj sequences for data mode: {self.data_mode} --')

            if self.data_mode == 'train':
                vessel_inds = self.train_ind
            elif self.data_mode == 'all':
                vessel_inds = self.traj_ind
            
            
            train_num = 0
            for t,ti in enumerate(vessel_inds):
#                 print('trajectory index ', t)
                df = self.dataset[self.dataset['trajectory']==ti]
                df_sog = df.loc[:,self.variable]
                df_sog = pd.DataFrame(df_sog)
                
                # limit trajectory length
                if df_sog.shape[0] > self.traj_length:
                    df_sog = df_sog[0:self.traj_length]

                    # sog_diff = difference_method(df_sog)
                sog_stand = standardization(df_sog)
                
                if sog_stand.isnull().values.any() == True:
                    print(f'nan value after standarization, sample num {t}, veesel num {ti}')
                    continue

                x_train_stand_seq = create_sequences(sog_stand.values, self.seq_length)
                if train_num==0:
                    xx_train = x_train_stand_seq
#                         print('added a new sample', xx_train.shape)

                if train_num > 0:
                    xx_train = np.concatenate((xx_train,x_train_stand_seq),axis=0)
#                         print('added a new sample', xx_train.shape)
                train_num +=1
                print(train_num)
                if train_num == self.max_traj_num:
                    break

            print('num of training vessels taken for scaled seqs', train_num)
            print('num of all generated scaled sequences', xx_train.shape)


            self.xtrain = xx_train
            self.train_num = train_num


             
    def create_train_traj(self):
        ''' it returns a matrix with shape  traj_length * n_traj --- good for saliency maps''' 
        ''' fixed size trajectory '''
        ''' not considering time steps beyond traj_length point '''
        
        train_num = 0
        for t,ti in enumerate(self.traj_ind):
#             print('sample ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,self.attribute]
            df_sog = pd.DataFrame(df_sog)
            if df_sog.shape[0] > 10000:   
        #         print('trajectory shape',df_sog.shape[0])
                df_x = df_sog[0:self.traj_length]
                if df_x.isnull().values.any() == True:
                    print('nan-----------------')
                    continue
                if train_num==0:
                    xx_train = df_x
                    print('first train sample',xx_train.shape)

                if train_num > 0:
                    xx_train = np.concatenate((xx_train,df_x),axis=1)
#                     print('train concatenated',xx_train.shape)
                train_num +=1
#                 print('num ', train_num)

        print('num of training samples', train_num)
        
       
        self.xtrain = xx_train
        self.train_num = train_num   
        


    def create_train_sc_traj(self):
        ''' it returns a matrix with shape  traj_length * n_traj --- good for saliency maps''' 
        ''' fixed size trajectory '''
        ''' trajectories are scaled between 0 and 1 '''

        print(f'-- create scaled traj sequences for data mode: {self.data_mode} --')

        if self.data_mode == 'train':
            vessel_inds = self.train_ind
        elif self.data_mode == 'all':
            vessel_inds = self.traj_ind
            
        print(f'-- number of vessels with lengths greater than 1000: {len(vessel_inds)} --')
    
        train_num = 0
        for t,ti in enumerate(vessel_inds):
#             print('sample ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,self.attribute]
            df_sog = pd.DataFrame(df_sog)
            if df_sog.shape[0] > 10000: #5000:   
        #         print('trajectory shape',df_sog.shape[0])
                df_x = df_sog[0:self.traj_length]
                df_x_stand = standardization(df_x)

                if df_x_stand.isnull().values.any() == True:
                    print('nan afer stand --------------')
                    continue
                if train_num==0:
                    xx_train = df_x_stand

                if train_num > 0:
                    xx_train = np.concatenate((xx_train,df_x_stand),axis=1)
#                     print('train concatenated',xx_train.shape)
                train_num +=1
                if train_num == self.max_traj_num:
                    break

        print('num of training vessels with length greater than 10000 taken for scaled seqs', train_num)
        print('shape of all scaled sequences', xx_train.shape)



        self.xtrain = xx_train
        self.train_num = train_num   
