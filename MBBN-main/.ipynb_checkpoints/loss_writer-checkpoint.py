from torch.nn import MSELoss,L1Loss,BCELoss, BCEWithLogitsLoss, Sigmoid
from losses import Mask_Loss, Spatial_Difference_Loss
import csv
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from itertools import zip_longest
from metrics import Metrics
import torch

import wandb
import time

class Writer():
    """
    main class to handle logging the results, both to tensorboard and to a local csv file
    """
    def __init__(self,sets,val_threshold,**kwargs):
        self.register_args(**kwargs)
        self.register_losses(**kwargs)
        self.spatial_loss_factor = kwargs.get('spatial_loss_factor')
        self.create_score_folders()
        self.metrics = Metrics()
        self.current_metrics = {}
        self.sets = sets
                
        if self.target == 'reconstruction':
            self.sets = ['train']
        self.val_threshold = val_threshold
        self.total_train_steps = 0
        self.eval_iter = 0
        self.subject_accuracy = {}
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_dir, comment=self.experiment_title)
        for set in sets:
            setattr(self,'total_{}_loss_values'.format(set),[])
            setattr(self,'total_{}_loss_history'.format(set),[])
        for name, loss_dict in self.losses.items():
            if loss_dict['is_active']:
                for set in sets:
                    setattr(self, '{}_{}_loss_values'.format(name,set),[])
                    setattr(self, '{}_{}_loss_history'.format(name,set),[])

    def create_score_folders(self):
        self.tensorboard_dir = Path(os.path.join(self.log_dir, self.experiment_title))
        self.csv_path = os.path.join(self.experiment_folder, 'history')
        os.makedirs(self.csv_path, exist_ok=True)
        if self.task == 'fine_tune' or 'bert' or 'test':
            self.per_subject_predictions = os.path.join(self.experiment_folder, 'per_subject_predictions')
            os.makedirs(self.per_subject_predictions, exist_ok=True)

    def save_history_to_csv(self):
        rows = [getattr(self, x) for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)]
        column_names = tuple([x for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)])
        export_data = zip_longest(*rows, fillvalue='')
        with open(os.path.join(self.csv_path, 'full_scores.csv'), 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(column_names)
            wr.writerows(export_data)


    def loss_summary(self,lr):
        self.scalar_to_tensorboard('learning_rate',lr,self.total_train_steps)
        loss_d = self.append_total_to_losses()
        for name, loss_dict in loss_d.items():
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set # name : binary classification & total : total_train
                    values = getattr(self,title + '_loss_values')
                    if len(values) == 0:
                        continue
                    score = np.mean(values)
                    history = getattr(self,title + '_loss_history') # total_train_loss_history
                    history.append(score)
                    print('{}: {}'.format(title,score))
                    setattr(self,title + '_loss_history',history)
                    self.scalar_to_tensorboard(title,score)
    
    
    
#     def accuracy_summary(self, mid_epoch, mean, std):
#         pred_all_sets = {x: [] for x in self.sets}
#         truth_all_sets = {x: [] for x in self.sets}
#         metrics = {}

#         # Collect all predictions and truths for AUROC calculation
#         all_truth = []
#         all_pred = []

#         for subj_name, subj_dict in self.subject_accuracy.items():
#             if self.fine_tune_task == 'binary_classification':
#                 # Apply sigmoid activation to logits
#                 subj_dict['score'] = torch.sigmoid(subj_dict['score'].float())

#             # Aggregate scores for the subject
#             subj_pred = subj_dict['score'].mean().item()
#             subj_truth = subj_dict['truth'].item()
#             subj_mode = subj_dict['mode']  # train, val, test

#             # Store for per-set metrics
#             pred_all_sets[subj_mode].append(subj_pred)
#             truth_all_sets[subj_mode].append(subj_truth)

#             # Collect for overall AUROC
#             all_truth.append(subj_truth)
#             all_pred.append(subj_pred)

#             # Log per-subject predictions
#             with open(os.path.join(self.per_subject_predictions, f'iter_{self.eval_iter}.txt'), 'a+') as f:
#                 f.write(f'subject: {subj_name} ({subj_mode})\noutputs: {subj_pred:.4f} - truth: {subj_truth}\n')

#         # Compute metrics for each set (train, val, test)
#         for (name, pred), (_, truth) in zip(pred_all_sets.items(), truth_all_sets.items()):
#             if len(pred) == 0:
#                 continue

#             if self.fine_tune_task == 'regression':
#                 unnormalized_pred = [i * std + mean for i in pred]
#                 unnormalized_truth = [i * std + mean for i in truth]

#                 metrics[name + '_MAE'] = self.metrics.MAE(unnormalized_truth, unnormalized_pred)
#                 metrics[name + '_MSE'] = self.metrics.MSE(unnormalized_truth, unnormalized_pred)
#                 metrics[name + '_NMSE'] = self.metrics.NMSE(unnormalized_truth, unnormalized_pred)
#                 metrics[name + '_R2_score'] = self.metrics.R2_score(unnormalized_truth, unnormalized_pred)
#             else:
#                 metrics[name + '_Balanced_Accuracy'] = self.metrics.BAC(truth, [x > 0.5 for x in torch.Tensor(pred)])
#                 metrics[name + '_Regular_Accuracy'] = self.metrics.RAC(truth, [x > 0.5 for x in torch.Tensor(pred)])

#                 if len(set(truth)) < 2:
#                     print(f"Skipping AUROC for set {name}. Only one class present: {set(truth)}")
#                     metrics[name + '_AUROC'] = None
#                 else:
#                     metrics[name + '_AUROC'] = self.metrics.AUROC(truth, pred)

#         # Compute combined AUROC across all subjects
#         if len(set(all_truth)) < 2:
#             print(f"Skipping combined AUROC calculation. Only one class present in all_truth: {set(all_truth)}")
#         else:
#             metrics['Combined_AUROC'] = self.metrics.AUROC(all_truth, all_pred)

#         # Save metrics
#         self.current_metrics = metrics
#         for name, value in metrics.items():
#             if value is None:  # Skip logging NoneType metrics
#                 print(f"Skipping TensorBoard logging for {name} due to NoneType value.")
#                 continue

#             self.scalar_to_tensorboard(name, value)
#             if hasattr(self, name):
#                 l = getattr(self, name)
#                 l.append(value)
#                 setattr(self, name, l)
#             else:
#                 setattr(self, name, [value])
#             print(f'{name}: {value}')


    
    
    def accuracy_summary(self, mid_epoch, mean, std):
        pred_all_sets = {x:[] for x in self.sets}   # dictionary to store predictions
        truth_all_sets = {x:[] for x in self.sets}  # dictionary to store ground truth values
        metrics = {}
        for subj_name,subj_dict in self.subject_accuracy.items():  # per-subject prediction scores (score), ground truth labels (truth), and the set (mode) they belong to
            
            if self.fine_tune_task == 'binary_classification':
                
                ### DEBUG STATEMENT ###
                # print(f"scores before sigmoid: {subj_dict['score'].float()}")
                #######################
                
                subj_dict['score'] = torch.sigmoid(subj_dict['score'].float())
                
                ### DEBUG STATEMENT ###
                # print(f"scores after sigmoid: {subj_dict['score']}")
                #######################


            # subj_dict['score'] denotes the logits for sequences for a subject
            subj_pred = subj_dict['score'].mean().item() 
            subj_error = subj_dict['score'].std().item()

            subj_truth = subj_dict['truth'].item()
            subj_mode = subj_dict['mode'] # train, val, test
            
            
            ### DEBUG STATEMENT ###
            # print(f"subj_dict['score']: {subj_dict['score']}")
            # print(f"subj_dict['truth']: {subj_dict['truth']}")
            #######################

            with open(os.path.join(self.per_subject_predictions,'iter_{}.txt'.format(self.eval_iter)),'a+') as f:
                f.write('subject:{} ({})\noutputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_pred,subj_error,subj_truth))
            pred_all_sets[subj_mode].append(subj_pred) # don't use std in computing AUROC, ACC
            truth_all_sets[subj_mode].append(subj_truth)
            
            ### DEBUG STATEMENT ###
            # print(f"pred_all_sets AFTER adding {subj_name}, subj_mode = {subj_mode}: {pred_all_sets}")
            # print(f"truth_all_sets AFTER adding {subj_name}, subj_mode = {subj_mode}: {truth_all_sets}")
            #######################
            

        for (name,pred),(_,truth) in zip(pred_all_sets.items(),truth_all_sets.items()):
            if len(pred) == 0:
                continue
            
            ### DEBUG STATEMENT ###
            # print(f"pred: {pred}")
            # print(f"truth: {truth}")
            #######################
            
            if self.fine_tune_task == 'regression':
                ## return to original scale ##
                unnormalized_pred = [i * std + mean for i in pred]
                unnormalized_truth = [i * std + mean for i in truth]

                metrics[name + '_MAE'] = self.metrics.MAE(unnormalized_truth,unnormalized_pred)
                metrics[name + '_MSE'] = self.metrics.MSE(unnormalized_truth,unnormalized_pred)
                metrics[name +'_NMSE'] = self.metrics.NMSE(unnormalized_truth,unnormalized_pred)
                metrics[name + '_R2_score'] = self.metrics.R2_score(unnormalized_truth,unnormalized_pred)
                
            else:
                metrics[name + '_Balanced_Accuracy'] = self.metrics.BAC(truth,[x>0.5 for x in torch.Tensor(pred)])
                metrics[name + '_Regular_Accuracy'] = self.metrics.RAC(truth,[x>0.5 for x in torch.Tensor(pred)]) # Stella modified it
                                
                ### DEBUG STATEMENT ###
                # print(f"Truth labels (y_true): {truth}")
                # print(f"Predicted scores (y_pred): {pred}")
                # print(f"Predicted classes: {[1 if p > 0.5 else 0 for p in pred]}")
                # print(f"Unique labels in y_true: {set(truth)}")
                # if len(truth) != len(pred):
                #     print(f"Length mismatch! y_true: {len(truth)}, y_pred: {len(pred)}")
                # if len(set(truth)) < 2:
                #     print("Skipping AUROC calculation. Only one class present in y_true.")
                #     return None
                #######################
                
                metrics[name + '_AUROC'] = self.metrics.AUROC(truth,pred)
                metrics[name +'_best_bal_acc'], metrics[name + '_best_threshold'],metrics[name + '_gmean'],metrics[name + '_specificity'],metrics[name + '_sensitivity'],metrics[name + '_f1_score'] = self.metrics.ROC_CURVE(truth,pred,name,self.val_threshold)
            self.current_metrics = metrics
            
            
        for name,value in metrics.items():
            self.scalar_to_tensorboard(name,value)
            if hasattr(self,name):
                l = getattr(self,name)
                l.append(value)
                setattr(self,name,l)
            else:
                setattr(self, name, [value])
            print('{}: {}'.format(name,value))
        self.eval_iter += 1
        if mid_epoch and len(self.subject_accuracy) > 0:
            self.subject_accuracy = {k: v for k, v in self.subject_accuracy.items() if v['mode'] == 'train'}
        else:
            self.subject_accuracy = {}

    def register_wandb(self,epoch, lr):
        wandb_result = {}
        wandb_result['epoch'] = epoch
        wandb_result['learning_rate'] = lr

        #losses 
        loss_d = self.append_total_to_losses() 
        for name, loss_dict in loss_d.items():
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set
                    wandb_result[f'{title}_loss_history'] = getattr(self,title + '_loss_history')[-1]
        #accuracy
        wandb_result.update(self.current_metrics)
        wandb.log(wandb_result)

    def write_losses(self,final_loss_dict,set):
        for loss_name,loss_value in final_loss_dict.items():
            title = loss_name + '_' + set
            loss_values_list = getattr(self,title + '_loss_values') # total_train_loss_values
            loss_values_list.append(loss_value)
            
#             if set == 'train':
#                 loss_values_list = loss_values_list[-self.running_mean_size:]
            setattr(self,title + '_loss_values',loss_values_list) # total_train_loss_values

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def register_losses(self,**kwargs):
        self.losses = {'reconstruction':
                           {'is_active':False,'criterion' : L1Loss(),'factor': 1},
                       'mask':
                           {'is_active':False,'criterion': Mask_Loss(**kwargs),'factor':1},
                       'binary_classification':
                           {'is_active':False,'criterion': BCEWithLogitsLoss(),'factor':1},
                       'regression':
                           {'is_active':False,'criterion':L1Loss(),'factor':1},
                      'spatial_difference':
                           {'is_active':False,'criterion':Spatial_Difference_Loss(**kwargs),'factor':self.spatial_loss_factor}}  #changed from L1Loss to MSELoss and changed to L1loss again
        print(kwargs.get('task').lower())
        if 'reconstruction' in kwargs.get('task').lower():
            if kwargs.get('use_recon_loss'):
                self.losses['reconstruction']['is_active'] = True
            if kwargs.get('use_mask_loss'):
                self.losses['mask']['is_active'] = True
            if kwargs.get('spatiotemporal'):
                self.losses['spatial_difference']['is_active'] = True
        else:
            if kwargs.get('fine_tune_task').lower() == 'regression':
                self.losses['regression']['is_active'] = True
            else:
                self.losses['binary_classification']['is_active'] = True 
            if kwargs.get('spatiotemporal'):
                self.losses['spatial_difference']['is_active'] = True

    def append_total_to_losses(self):
        loss_d = self.losses.copy()
        loss_d.update({'total': {'is_active': True}})
        return loss_d

    def scalar_to_tensorboard(self,tag,scalar,iter=None):
        if iter is None:
            iter = self.total_train_steps if 'train' in tag else self.eval_iter
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag,scalar,iter)

