import numpy as np
import torch
from torch.utils.data import DataLoader,Subset, Dataset, RandomSampler
from pathlib import Path
#DDP
from torch.utils.data.distributed import DistributedSampler
from data_preprocess_and_load.datasets import *
from utils import reproducibility
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
import pandas as pd

##### ADDED #####
# # Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
################# 


class DataHandler():  # primary class for dataset management (initialization, preprocessing, split generation, loader creation)
    def __init__(self,test=False,**kwargs): # sets up the handler with used-provided configurations
        self.step = kwargs.get('step')  # processing step
        self.base_path = kwargs.get('base_path')  # the base directory for the dataset
        self.kwargs = kwargs  # keyword arguments
        self.dataset_name = kwargs.get('dataset_name') 
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')  # task type (e.g., binary classification, regression)
        self.finetune = kwargs.get('finetune')  ## ADDED
        self.seq_len = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')  # dimentionality of intermediate vectors (e.g., number of ROIs)
        self.seed = kwargs.get('seed')  # random seed for reproducibility
        self.fmri_dividing_type = kwargs.get('fmri_dividing_type')  # ADDED
        self.fmri_type = kwargs.get('fmri_type')  # ADDED
        reproducibility(**self.kwargs)
        dataset = self.get_dataset()
        self.train_dataset = dataset(**self.kwargs)  # initializes the training dataset
        self.eval_dataset = dataset(**self.kwargs)  # initializes the evaluation dataset
        if self.fine_tune_task == 'regression':  # sets mean and SD for regression tasks
            if self.step != '3':
                self.mean = self.train_dataset.mean
                self.std = self.train_dataset.std
            elif self.step == '3':
                self.mean = self.eval_dataset.mean
                self.std = self.eval_dataset.std
        self.eval_dataset.augment = None  # disables data augmentation for evaluation datasets
        
        if self.target == 'ADHD_label':  # standardizing target names
            self.target = 'ADHD'
        elif self.target == 'ASD':
            self.target == 'DX_GROUP'
        elif self.target == 'nihtbx_totalcomp_uncorrected':
            self.target = 'total_intelligence'
        elif self.target == 'ASD_label':
            if self.dataset_name == 'ABCD':
                self.target = 'ASD'
        elif self.target == 'OCD':  # ENIGMA-OCD target
            self.target = 'OCD'
        

        self.splits_folder = Path(self.base_path).joinpath('splits',self.dataset_name)  # the directory for saving/loading data splits
        self.current_split = self.splits_folder.joinpath(f"{self.dataset_name}_{self.target}_ROI_{self.intermediate_vec}_seq_len_{self.seq_len}_split{self.seed}.txt")  # file path for the current split
        
        if not self.current_split.exists():  # handles missing splits
            print('generating splits...')
            
            # go back to original target in metadata
            if self.target == 'ADHD':
                self.target = 'ADHD_label'
            elif self.target == 'total_intelligence':
                self.target = 'nihtbx_totalcomp_uncorrected'
            elif self.target == 'ASD':
                if self.dataset_name == 'ABCD':
                    self.target = 'ASD_label' 
            elif self.target == 'depression':
                self.target = 'MDD_pp'
            elif self.target == 'OCD':  # ENIGMA-OCD target
                self.target = 'OCD'
                
            ## generate stratified sampler
            if self.dataset_name == 'ABCD':  # reading meta-data
                sub = [i.split('-')[1] for i in os.listdir(kwargs.get('abcd_path'))]
                if self.target == 'MDD_pp':
                    metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_5_1_KSADS_raw_MDD_ANX_CorP_pp_pres_ALL.csv'))
                    metadata['subjectkey'] = [i.split('-')[1] for i in metadata['subjectkey']]
                else:
                    metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABCD_phenotype_total.csv'))
            elif self.dataset_name == 'UKB':
                sub = [str(i) for i in os.listdir(kwargs.get('ukb_path'))]
                metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/UKB_phenotype_gps_fluidint.csv'))
            elif self.dataset_name == 'ABIDE':
                sub = os.listdir(kwargs.get('abide_path'))                    
                metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ABIDE1+2_meta.csv'))
            elif self.dataset_name == 'ENIGMA_OCD':
                sub = os.listdir(kwargs.get('enigma_path'))                    
                metadata = pd.read_csv(os.path.join(self.base_path, './data/metadata/ENIGMA_QC_final_subject_list.csv'))
                
            if self.target == 'SuicideIdeationtoAttempt':
                new_meta = metadata[['subjectkey', 'sex', self.target]].dropna()
                        
            elif self.target == 'reconstruction':  # no prediction task
                new_meta = None

            else:
                if self.dataset_name == 'ABCD':
                    new_meta = metadata[['subjectkey', self.target]].dropna()
                elif self.dataset_name == 'UKB':
                    new_meta = metadata[['eid', self.target]].dropna()
                    new_meta['eid'] = new_meta['eid'].astype('object')
                elif self.dataset_name == 'ABIDE':
                    new_meta = metadata[['SUB_ID', 'DX_GROUP']].dropna()
                elif self.dataset_name == 'ENIGMA_OCD':
                    new_meta = metadata[['Unique_ID', 'OCD']].dropna()

            # 01 remove subjects which have NaN voxel from the original 4D data
            print('generating step 1')
            valid_sub = []
            prob_sub = []
            for i in sub:
                if self.dataset_name == 'ABCD':
                    if self.intermediate_vec == 180:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'hcp_mmp1_180_sub-'+i+'.npy')
                    elif self.intermediate_vec == 360:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'hcp_mmp1_sub-'+i+'.npy')
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('abcd_path'), 'sub-'+i+'/'+'schaefer_sub-'+i+'.npy')
                    file = np.load(filename)[:self.seq_len].T
                elif self.dataset_name == 'UKB':
                    if self.intermediate_vec == 180:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'hcp_mmp1_'+i+'.npy')
                    elif self.intermediate_vec == 400:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    elif self.intermediate_vec == 304:
                        filename = os.path.join(kwargs.get('ukb_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    file = np.load(filename)[20:20+self.seq_len].T
                elif self.dataset_name == 'ABIDE':
                    ## only have schaefer atlas due to the /storage problem .. :(
                    filename = os.path.join(kwargs.get('abide_path'), i+'/'+'schaefer_400Parcels_17Networks_'+i+'.npy')
                    file = np.load(filename)[20:20+self.seq_len].T
                elif self.dataset_name == 'ENIGMA_OCD':
                    if self.intermediate_vec == 316:
                        if self.fmri_type in {"divided_timeseries", "timeseries"}:
                            # filename = os.path.join(kwargs.get('enigma_path'), i+'/'+i+'_filtered_0.01_0.1.npy')  # band-pass-filtered data
                            filename = os.path.join(kwargs.get('enigma_path'), i+'/'+i+'.npy')  # unfiltered data
                        else:
                            raise ValueError("Filename is not defined for this fMRI type")

                        file = np.load(filename)[20:20+self.seq_len].T 
                        site = filename.split('/')[-2].split('_')[-2]      

                    
                ##### ADDED #####
                # if 'Amsterdam-AMC' in site:
                #     TR = 2.375
                # elif 'Amsterdam-VUmc' in site:
                #     TR = 1.8
                # elif 'Barcelona-HCPB' in site:
                #     TR = 2
                # elif 'Bergen' in site:
                #     TR = 1.8
                # elif 'Braga-UMinho-Braga-1.5T' in site:
                #     TR = 2
                # elif 'Braga-UMinho-Braga-1.5T-act' in site:
                #     TR = 2
                # elif 'Braga-UMinho-Braga-3T' in site:
                #     TR = 1
                # elif 'Brazil' in site:
                #     TR = 2
                # elif 'Cape-Town-UCT-Allegra' in site:
                #     TR = 1.6
                # elif 'Cape-Town-UCT-Skyra' in site:
                #     TR = 1.73
                # elif 'Chiba-CHB' in site:
                #     TR = 2.3
                # elif 'Chiba-CHBC' in site:
                #     TR = 2.3 
                # elif 'Chiba-CHBSRPB' in site:
                #     TR = 2.5 
                # elif 'Dresden' in site:
                #     TR = 0.8 
                # elif 'Kyoto-KPU-Kyoto1.5T' in site:
                #     TR = 2.411 
                # elif 'Kyoto-KPU-Kyoto3T' in site:
                #     TR = 2
                # elif 'Kyushu' in site:
                #     TR = 2.5
                # elif 'Milan-HSR' in site:
                #     TR = 2
                # elif 'New-York' in site:
                #     TR = 1
                # elif 'NYSPI-Columbia-Adults' in site:
                #     TR = 0.85
                # elif 'NYSPI-Columbia-Pediatric' in site:
                #     TR = 0.85
                # elif 'Yale-Pittinger-HCP-Prisma' in site:
                #     TR = 0.8
                # elif 'Yale-Pittinger-HCP-Trio' in site:
                #     TR = 0.7
                # elif 'Yale-Pittinger-Yale-2014' in site:
                #     TR = 2
                # elif 'Bangalore-NIMHANS' in site:
                #     TR = 2 
                # elif 'Barcelone-Bellvitge-ANTIGA-1.5T' in site:
                #     TR = 2
                # elif 'Barcelone-Bellvitge-COMPULSE-3T' in site:
                #     TR = 2
                # elif 'Barcelone-Bellvitge-PROV-1.5T' in site:
                #     TR = 2
                # elif 'Barcelone-Bellvitge-RESP-CBT-3T' in site:
                #     TR = 2
                # elif 'Seoul-SNU' in site:
                #     TR = 3.5
                # elif 'Shanghai-SMCH' in site:
                #     TR = 3
                # elif 'UCLA' in site:
                #     TR = 2
                # elif 'Vancouver-BCCHR' in site:
                #     TR = 2
                # elif 'Yale-Gruner' in site:
                #     TR = 2
                # else:
                #     raise ValueError(f"Site '{site}' does not have a defined TR value in TR_mappings. Please add it.")
                
                # if file.shape[1] >= self.seq_len:
                #     sample_whole = np.zeros(self.seq_len,) # originally self.sequence_length   ## aggregates time-series data across ROIs

                #     for l in range(self.intermediate_vec):
                #         sample_whole+=file[l]
                        
                #     sample_whole /= self.intermediate_vec    # averages the time-series signals (y) across a set number of ROIs

                #     T = TimeSeries(sample_whole, sampling_interval=TR)  # computes power spectral density (PSD) of the averaged time-series signal
                #     S_original = SpectralAnalyzer(T)

                #     # Lorentzian function fitting (dividing ultralow ~ low)  ## extracts the PSD data
                #     xdata = np.array(S_original.spectrum_fourier[0][1:])  # xdata = frequency values  
                #     ydata = np.abs(S_original.spectrum_fourier[1][1:])    # ydata = corresponding power values

                #     # initial parameter setting
                #     p0 = [0, 0.006]   
                #     param_bounds = ([-np.inf, 0], [np.inf, 1])

                #     # fitting Lorentzian function
                #     popt, pcov = curve_fit(lorentzian_function, xdata, ydata, p0=p0, maxfev = 5000, bounds=param_bounds)   # popt = optimal parameters
                #     f1 = popt[1]
                #     knee = round(popt[1]/(1/(sample_whole.shape[0]*TR)))   # calculates knee frequency
                    
                #     # Vanilla BERT does not require frequency band division
                #     if self.step == 1:
                #         valid_sub.append(i)
                #     elif self.fmri_dividing_type == 'four_channels':
                #         valid_sub.append(i)
                #     else:
                #         if knee < xdata.shape[0] and knee < ydata.shape[0]:
                #             valid_sub.append(i)

                # for j in range(file.shape[0]):  # checks for zero ROIs
                #     if np.sum(file[j]) == 0:  # if all time points for a specific ROI are zero, the subject ID is added to the problematic subject list
                #         prob_sub.append(i)
                #################

                """
                Sequence length padding experiment
                """    
                valid_sub.append(i)
                """
                """
               
            valid_sub = list(set(valid_sub) - set(prob_sub))  # removes problematic subjects
            print(f"Number of subjects used for training: {len(valid_sub)}")
                
            if self.dataset_name == 'UKB':
                valid_sub = list(map(int, valid_sub))  # converts subject IDs into integers
                        
            # 02 select subjects with target and split file
            print('generating step 2')
            if self.target == 'reconstruction':
                sublist = ['train_subjects']+list(valid_sub)+['val_subjects']+['test_subjects']+[' ']
            else:
                if self.dataset_name == 'ABCD':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'subjectkey'})  # convert the list of valid subject IDs into a pandas DataFrame with the 'subjectkey' column
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='subjectkey')  # merge valid subject IDs with the meta-data (keep only the valid subjects)

                    if self.target == 'SuicideIdeationtoAttempt':
                        '''stratified sampling for two columns'''
                        X_train, X_test = train_test_split(new_meta['subjectkey'],
                                          test_size=0.15,
                                          stratify= new_meta[['sex', self.target]],  #### STRATIFIED SAMPLING ON BOTH 'SEX' AND 'SUICIDAL IDEATION TO ATTEMPT'
                                          random_state = self.seed)

                        train_and_valid = new_meta[new_meta['subjectkey'].isin(X_train)]

                        X_train, X_valid = train_test_split(train_and_valid['subjectkey'],
                                                          test_size=0.175,
                                                          stratify= train_and_valid[['sex', self.target]],
                                                          random_state = self.seed)
                    else:
                        if self.fine_tune_task == 'binary_classification':
                            X_train, X_test, y_train, y_test = train_test_split(new_meta['subjectkey'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,    # test set is 15% of the total dataset
                                                                    stratify= new_meta[self.target],
                                                                    random_state = self.seed)

                            X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,   # validation set is 17.5% of the remaining 85% (total dataset - test set)
                                                                              stratify= y_train,
                                                                              random_state = self.seed)
                        elif self.fine_tune_task == 'regression':
                            X_train, X_test, y_train, y_test = train_test_split(new_meta['subjectkey'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    random_state = self.seed)

                            X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              random_state = self.seed)
                            
                            
                elif self.dataset_name == 'UKB':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'eid'})
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='eid')
                    if self.fine_tune_task == 'binary_classification':
                        X_train, X_test, y_train, y_test = train_test_split(new_meta['eid'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    stratify= new_meta[self.target],
                                                                    random_state = self.seed)

                        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              stratify= y_train,
                                                                              random_state = self.seed)
                    elif self.fine_tune_task == 'regression':
                        X_train, X_test, y_train, y_test = train_test_split(new_meta['eid'],
                                                                new_meta[self.target],
                                                                test_size=0.15,
                                                                random_state = self.seed)

                        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                          y_train,
                                                                          test_size=0.175,
                                                                          random_state = self.seed)
                elif self.dataset_name == 'ABIDE':
                    subid = valid_sub # [i[2:] if i.startswith('00') else i for i in valid_sub]
                    # starts with 5 or 2 now! (because metadata doesn't starts with 00)

                    valid_df = pd.DataFrame(subid).rename(columns = {0 : 'SUB_ID'}) # ['SUB_ID']
                    new_meta = new_meta.rename(columns= {'DX_GROUP': 'ASD'}) # ['SUB_ID', 'ASD'] # 두 dataframe 다 int.

                    new_meta['SUB_ID'] = new_meta['SUB_ID'].astype(str)
                    new_meta['SUB_ID'] = ['00' + i if '00' + i in subid else i for i in new_meta['SUB_ID']]
                    # now new_meta's SUB_ID contains 00
                    
                    valid_df['SUB_ID'] = valid_df['SUB_ID'].astype(str)
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='SUB_ID')
                    
                    if self.target == 'DX_GROUP':
                        self.target = 'ASD'
                    X_train, X_test, y_train, y_test = train_test_split(new_meta['SUB_ID'],
                                                                new_meta[self.target],
                                                                test_size=0.15,
                                                                stratify= new_meta[self.target],
                                                                random_state = self.seed)

                    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                          y_train,
                                                                          test_size=0.175,
                                                                          stratify= y_train,
                                                                          random_state = self.seed)
                    
                elif self.dataset_name == 'ENIGMA_OCD':
                    valid_df = pd.DataFrame(valid_sub).rename(columns = {0 : 'Unique_ID'})
                    new_meta = pd.merge(new_meta, valid_df, how = 'inner', on='Unique_ID')

                    if self.target == 'OCD' and self.fine_tune_task == 'binary_classification':
                        X_train, X_test, y_train, y_test = train_test_split(new_meta['Unique_ID'],
                                                                    new_meta[self.target],
                                                                    test_size=0.15,
                                                                    stratify= new_meta[self.target],
                                                                    random_state = self.seed)

                        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.175,
                                                                              stratify= y_train,
                                                                              random_state = self.seed)
                
                    
                    ### DEBUG STATEMENT ###
                    # Analyze class distribution
                    train_labels = y_train.tolist()
                    val_labels = y_valid.tolist()
                    test_labels = y_test.tolist()

                    print(f"Training set class distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
                    print(f"Validation set class distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
                    print(f"Test set class distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
                    #######################
                
                sublist = ['train_subjects']+list(X_train)+['val_subjects']+list(X_valid)+['test_subjects']+list(X_test)   # a formatted list to store the train, validation, and test subject IDs
                # ['train_subjects', 'sub-001', 'sub-002', 'sub-003',  # Training subjects
                #  'val_subjects', 'sub-004', 'sub-005',             # Validation subjects
                #  'test_subjects', 'sub-006', 'sub-007' ]             # Test subjects 

                if self.target == 'ADHD_label':   # target name standardization
                    self.target = 'ADHD'
                elif self.target == 'nihtbx_totalcomp_uncorrected':
                    self.target = 'total_intelligence'
                elif self.target == 'ASD_label':
                    self.target = 'ASD'
                elif self.target == 'MDD_pp':
                    self.target = 'depression'
                elif self.target == 'OCD':
                    self.target = 'OCD'
            
            if self.dataset_name == 'UKB':   # convert all subject IDs to strings, as some might initially be integers
                sublist = list(map(str, sublist))
            elif self.dataset_name == 'ENIGMA_OCD':   
                sublist = list(map(str, sublist))
                
            print('generating step 3.. saving splits...')   # save the subject splits to a file
            with open(f"./splits/{self.dataset_name}/{self.dataset_name}_{self.target}_ROI_{self.intermediate_vec}_seq_len_{self.seq_len}_split{self.seed}.txt", mode="w") as file:
                file.write('\n'.join(sublist))   # ex) ./splits/ENIGMA_OCD/ENIGMA_OCD_OCD_ROI_400_seq_len_280_split1.txt

        print(self.current_split)
        
    def get_mean_std(self):
        return None
    
    def get_dataset(self):  # returns the appropriate dataset class based on the dataset name
        if self.dataset_name == 'ABCD':
            return ABCD_fMRI_timeseries
        elif self.dataset_name == 'HCP1200':
            return HCP_fMRI_timeseries
        elif self.dataset_name == 'ABIDE':
            return ABIDE_fMRI_timeseries
        elif self.dataset_name == 'UKB':
            return UKB_fMRI_timeseries
        elif self.dataset_name == 'ENIGMA_OCD':
            return ENIGMA_OCD_fMRI_timeseries
        
    def current_split_exists(self):
        return self.current_split.exists()   # checks whether a pre-saved dataset split file exists


    def create_dataloaders(self):  # loads split data into PyTorch DataLoaders for training, validation, and testing
        reproducibility(**self.kwargs) 

        subject = open(self.current_split, 'r').readlines()   # load subject splits
        subject = [x[:-1] for x in subject]
        subject.remove('train_subjects')
        subject.remove('val_subjects')
        subject.remove('test_subjects')
        self.subject_list = self.train_dataset.index_l
        
        
        if self.current_split_exists():
            print('loading splits')
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names,self.subject_list)   # converts the subject IDs into indices for PyTorch's Subset class
        

        print('length of train_idx:', len(train_idx))
        print('length of val_idx:', len(val_idx))
        print('length of test_idx:', len(test_idx))
        
        train_dataset = Subset(self.train_dataset, train_idx)   # creates train, validation, and test datasets
        val_dataset = Subset(self.eval_dataset, val_idx)
        test_dataset = Subset(self.eval_dataset, test_idx)
        
        if self.kwargs.get('distributed'):  # distributed training (data distribution across multiple GPUs)
            print('distributed')
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            print('length of train sampler is:', len(train_sampler)) 
            if self.target != 'reconstruction':   
                valid_sampler = DistributedSampler(val_dataset, shuffle=False)
                print('length of valid sampler is:', len(valid_sampler)) 
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                print('length of test sampler is:', len(test_sampler))
        else:
            train_sampler = RandomSampler(train_dataset)  # random sampling for standalone training
            if self.target != 'reconstruction':
                valid_sampler = RandomSampler(val_dataset)
                test_sampler = RandomSampler(test_dataset)
        
        ## Stella transformed this part ##
        training_generator = DataLoader(train_dataset, **self.get_params(**self.kwargs),
                                       sampler=train_sampler)
        print('length of training generator is:', len(training_generator))
        
        if self.target != 'reconstruction':  # returns DataLoaders
            val_generator = DataLoader(val_dataset, **self.get_params(eval=True,**self.kwargs),
                                      sampler=valid_sampler)
            print('length of valid generator is:', len(val_generator))
            
            test_generator = DataLoader(test_dataset, **self.get_params(eval=True,**self.kwargs),
                               sampler=test_sampler)
            print('length of test generator is:', len(test_generator))
        
        
        else:   # if self.target == 'reconstruction', validation and test DataLoaders are skipped
            val_generator = None
            test_generator = None
        
        
        # ### DEBUG STATEMENT ###
        # print(f"Number of training batches: {len(training_generator)}")
        # print(f"Number of validation batches: {len(val_generator)}")
        # print(f"Number of test batches: {len(test_generator)}")
        # #######################
        
                   
        if self.fine_tune_task == 'regression':
            return training_generator, val_generator, test_generator, self.mean, self.std
            
        else:
            return training_generator, val_generator, test_generator   # return DataLoaders
    
    
    def get_params(self,eval=False,**kwargs):  # provides parameters for DataLoader configuration
        batch_size = kwargs.get('batch_size')
        workers = kwargs.get('workers')   # number of parallel workers for data loading
        cuda = kwargs.get('cuda')
        
        ### DEBUG STATEMENT ###
        print(f"workers: {workers}")
        #######################
        #if eval:
        #    workers = 0
        params = {'batch_size': batch_size,
                  #'shuffle': True,
                  'num_workers': workers,
                  'drop_last': True,   # drops the last incomplete batch if its size is smaller than batch_size
                  'pin_memory': True,  # True if cuda else False,
                  'persistent_workers': True if workers > 0 and cuda else False}  # keeps workers alive between batches when using multiple workers (improves efficiency)
        
        return params

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names,subj_list):  # maps subject IDs to dataset indices for split assignment
        subj_idx = np.array([str(x[1]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        
        return train_idx,val_idx,test_idx
    
    def load_split(self):  # loads pre-saved dataset splits into train, validation, and test groups
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train' in line for line in subject_order])   # finds where each split starts in the file
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index] # NDAR~ 형태  ## from train_index + 1 to val_index
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
              
        return train_names,val_names,test_names
