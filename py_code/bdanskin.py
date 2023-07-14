# -*- coding: utf-8 -*-
"""
Updated June 7 2023

@author: Bethanny Danskin
"""
import numpy as np
import pickle
import random
import xarray as xr
import pandas as pd
import warnings
# Load optimizers
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize, basinhopping
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

## Data directory
def get_sessions(area): #
    if area=='RSC':
        sessions_all = ['RH055_180715_RSC_l','RH055_180807_RSC_r',
                        'RH548_170309_RSC_l','RH730_170504_RSC_r',
                        'RH730_170510_RSC_l','RH731_170514_RSC_r',
                        'RH731_170527_RSC_l','RH795_170804_RSC_l',
                        'RH795_170808_RSC_r','RH824_170811_RSC_l',
                        'RH824_170814_RSC_r','RH825_171026_RSC_r',
                        'RH825_171028_RSC_l','RH883_180123_RSC_l',
                        'RH883_180125_RSC_r']
    elif area=='V1':
        sessions_all = ['RH054_180416_V1_l', 'RH055_180709_V1_r',
                        'RH795_170902_V1_r', 'RH795_170904_V1_l',
                        'RH824_170825_V1_r', 'RH824_170826_V1_l']
    elif area=='PPC':
        sessions_all = ['RH253_160725_PPC_r', 'RH289_160915_PPC_r',
                        'RH290_161005_PPC_r', 'RH548_170316_PPC_l',
                        'RH652_170215_PPC_l', 'RH730_170425_PPC_r',
                        'RH730_170426_PPC_l', 'RH731_170512_PPC_l', 
                        'RH731_170520_PPC_r', 'RH795_170806_PPC_l', 
                        'RH795_170807_PPC_r', 'RH824_170812_PPC_r', 
                        'RH824_170822_PPC_l', 'RH825_171104_PPC_l', 
                        'RH883_180116_PPC_r', 'RH883_180121_PPC_l']
    elif area=='ALM':

        sessions_all =  ['RH052_180311_ALM_l', 'RH054_180414_ALM_l',
                         'RH054_180417_ALM_r', 'RH055_180712_ALM_r',
                         'RH795_170829_ALM_r', 'RH795_170901_ALM_l',
                         'RH824_170824_ALM_l', 'RH824_170827_ALM_r',
                         'RH825_171019_ALM_l', 'RH825_171022_ALM_r',
                         'RH871_171008_ALM_r', 'RH871_171015_ALM_l']
    elif area=='M2':
        sessions_all =  ['RH052_180319_M2_r', 'RH052_180320_M2_l', 
                         'RH054_180418_M2_l', 'RH054_180419_M2_r', 
                         'RH055_180711_M2_l', 'RH055_180806_M2_r', 
                         'RH652_170220_M2_l', 'RH730_170502_M2_r', 
                         'RH730_170511_M2_l', 'RH731_170519_M2_r', 
                         'RH731_170526_M2_l', 'RH795_170905_M2_r',
                         'RH825_171021_M2_r', 'RH871_171013_M2_r', 
                         'RH871_171014_M2_l', 'RH883_180106_M2_l',
                         'RH883_180108_M2_r']
    elif area=='S1':
        sessions_all =  ['RH052_180318_S1_r', 'RH052_180323_S1_l', 
                        'RH731_170521_S1_r','RH731_170523_S1_l',
                        'RH795_170815_S1_r', 'RH795_170820_S1_l',
                        'RH824_170817_S1_r', 'RH824_170818_S1_l',
                        'RH825_171103_S1_r', 'RH825_171105_S1_l',
                        'RH871_171017_S1_r', 'RH871_171018_S1_l',
                        'RH883_180124_S1_r', 'RH883_180126_S1_l']
    elif area=='all':
        sessions_all = ['RH055_180715_RSC_l','RH055_180807_RSC_r', # RSC
                        'RH548_170309_RSC_l','RH730_170504_RSC_r',
                        'RH730_170510_RSC_l','RH731_170514_RSC_r',
                        'RH731_170527_RSC_l','RH795_170804_RSC_l',
                        'RH795_170808_RSC_r','RH824_170811_RSC_l',
                        'RH824_170814_RSC_r','RH825_171026_RSC_r',
                        'RH825_171028_RSC_l','RH883_180123_RSC_l',
                        'RH883_180125_RSC_r',
                        # 'RH054_180416_V1_l', 'RH055_180709_V1_r', # V1
                        # 'RH795_170902_V1_r', 'RH795_170904_V1_l',
                        # 'RH824_170825_V1_r', 'RH824_170826_V1_l',
                        'RH253_160725_PPC_r', 'RH289_160915_PPC_r', # PPC
                        'RH290_161005_PPC_r', 'RH548_170316_PPC_l', 
                        'RH652_170215_PPC_l', 'RH730_170425_PPC_r', 
                        'RH730_170426_PPC_l', 'RH731_170512_PPC_l', 
                        'RH731_170520_PPC_r', 'RH795_170806_PPC_l', 
                        'RH795_170807_PPC_r', 'RH824_170812_PPC_r', 
                        'RH824_170822_PPC_l', 'RH825_171104_PPC_l', 
                        'RH883_180116_PPC_r', 'RH883_180121_PPC_l',
                        'RH052_180311_ALM_l', 'RH054_180414_ALM_l', # ALM
                        'RH054_180417_ALM_r', 'RH055_180712_ALM_r',
                        'RH795_170829_ALM_r', 'RH795_170901_ALM_l',
                        'RH824_170824_ALM_l', 'RH824_170827_ALM_r',
                        'RH825_171019_ALM_l', 'RH825_171022_ALM_r',
                        'RH871_171008_ALM_r', 'RH871_171015_ALM_l',
                        'RH052_180319_M2_r', 'RH052_180320_M2_l', # M2
                        'RH054_180418_M2_l', 'RH054_180419_M2_r', 
                        'RH055_180711_M2_l', 'RH055_180806_M2_r', 
                        'RH652_170220_M2_l', 'RH730_170502_M2_r', 
                        'RH730_170511_M2_l', 'RH731_170519_M2_r', 
                        'RH731_170526_M2_l', 'RH795_170905_M2_r',
                        'RH825_171021_M2_r', 'RH871_171013_M2_r', 
                        'RH871_171014_M2_l', 'RH883_180106_M2_l',
                        'RH883_180108_M2_r',
                        'RH052_180318_S1_r', 'RH052_180323_S1_l', # S1
                        'RH731_170521_S1_r','RH731_170523_S1_l',
                        'RH795_170815_S1_r', 'RH795_170820_S1_l',
                        'RH824_170817_S1_r', 'RH824_170818_S1_l',
                        'RH825_171103_S1_r', 'RH825_171105_S1_l',
                        'RH871_171017_S1_r', 'RH871_171018_S1_l',
                        'RH883_180124_S1_r', 'RH883_180126_S1_l']
    elif area=='early_RSC':
        sessions_all = ['RH052_180214_RSCe_l',
                        'RH054_180219_RSCe_r', 'RH054_180218_RSCe_l',
                        'RH055_180422_RSCe_l',
                        'RH795_170721_RSCe_r', 'RH795_170720_RSCe_l',
                        'RH824_170721_RSCe_l', 'RH824_170720_RSCe_r',
                        'RH825_171008_RSCe_r', 'RH825_171007_RSCe_l',
                        'RH883_171118_RSCe_r', 'RH883_171117_RSCe_l',
                        'RH896_171224_RSCe_r', 'RH896_171223_RSCe_l']
    elif area=='early_PPC':
        sessions_all = ['RH055_180423_RSCe_r', 'RH253_160711_PPCe_r',
                        'RH289_160901_PPCe_r', 'RH290_160921_PPCe_r',
                        'RH548_170227_PPCe_l', 'RH652_170203_PPCe_l',
                        'RH730_170419_PPCe_r', 'RH730_170418_PPCe_l',
                        'RH731_170501_PPCe_l',
                        'RH795_170722_PPCe_l', 'RH795_170719_PPCe_r',
                        'RH824_170723_PPCe_l', 'RH824_170722_PPCe_r',
                        'RH825_171009_PPCe_l']
    elif area=='early_ALM':
        sessions_all = ['RH052_180213_ALMe_l',
                        'RH054_180217_ALMe_l', 'RH054_180214_ALMe_r',
                        'RH055_180420_ALMe_l', 'RH055_180417_ALMe_r',
                        'RH825_171005_ALMe_l', 'RH825_171004_ALMe_r',
                        'RH871_170921_ALMe_l', 'RH871_170918_ALMe_r',
                        'RH896_171222_ALMe_l']
    elif area=='early_M2':
        sessions_all = ['RH052_180215_M2e_l', 'RH052_180212_M2e_r',
                        'RH054_180216_M2e_r', 'RH054_180215_M2e_l',
                        'RH055_180421_M2e_l', 'RH055_180419_M2e_r', 
                        'RH825_171003_M2e_r',
                        'RH871_170922_M2e_r', 'RH871_170916_M2e_l',
                        'RH883_171116_M2e_l', 'RH883_171115_M2e_r',
                        'RH896_171225_M2e_l']
    elif area=='early_all':
        sessions_all = ['RH052_180214_RSCe_l',
                        'RH054_180219_RSCe_r', 'RH054_180218_RSCe_l',
                        'RH055_180422_RSCe_l',
                        'RH795_170721_RSCe_r', 'RH795_170720_RSCe_l',
                        'RH824_170721_RSCe_l', 'RH824_170720_RSCe_r',
                        'RH825_171008_RSCe_r', 'RH825_171007_RSCe_l',
                        'RH883_171118_RSCe_r', 'RH883_171117_RSCe_l',
                        'RH896_171224_RSCe_r', 'RH896_171223_RSCe_l',
                        'RH055_180423_RSCe_r', 'RH253_160711_PPCe_r',
                        'RH289_160901_PPCe_r', 'RH290_160921_PPCe_r',
                        'RH548_170227_PPCe_l', 'RH652_170203_PPCe_l',
                        'RH730_170419_PPCe_r', 'RH730_170418_PPCe_l',
                        'RH731_170501_PPCe_l',
                        'RH795_170722_PPCe_l', 'RH795_170719_PPCe_r',
                        'RH824_170723_PPCe_l', 'RH824_170722_PPCe_r',
                        'RH825_171009_PPCe_l',
                        'RH052_180213_ALMe_l',
                        'RH054_180217_ALMe_l', 'RH054_180214_ALMe_r',
                        'RH055_180420_ALMe_l', 'RH055_180417_ALMe_r',
                        'RH825_171005_ALMe_l', 'RH825_171004_ALMe_r',
                        'RH871_170921_ALMe_l', 'RH871_170918_ALMe_r',
                        'RH896_171222_ALMe_l',
                        'RH052_180215_M2e_l', 'RH052_180212_M2e_r',
                        'RH054_180216_M2e_r', 'RH054_180215_M2e_l',
                        'RH055_180421_M2e_l', 'RH055_180419_M2e_r', 
                        'RH825_171003_M2e_r',
                        'RH871_170922_M2e_r', 'RH871_170916_M2e_l',
                        'RH883_171116_M2e_l', 'RH883_171115_M2e_r',
                        'RH896_171225_M2e_l']   
    else:
        print('select appropriate area')
    return sessions_all

def plt_color_dir(): #
    plt_colors = {'exp': [0, .5, 0], 'hyp': [.8, .2, .8],
                  'opto': [0,0.35,.85], 'post': [.5,0.15,.75], 
                  'RSC': [0,.75,.75], 'PPC': [.15,0,1], 'ALM': [.75,.5, 0],
                  'M2': [0,.25, 0], 'S1': [.5,0,.75], 'V1': [0,1,1],
                  'ChR2': [0,0.35,.85], 'ChRM': [1, .5, .5]}
    return plt_colors

## General Processing Functions
def prepare_hist_matrix(a,R,n_back): # Now pads with zeros
     # Prepare history matrix
    num_trials = len(a)
    uR = np.zeros(num_trials)
    uR[(R==0) & (a<3)] = 1
    
    a_hist = np.zeros((num_trials,n_back)); # a_hist[:] = np.NaN
    R_hist = np.zeros((num_trials,n_back)); # R_hist[:] = np.NaN
    uR_hist = np.zeros((num_trials,n_back))

    a_cleaned = a.copy()+0.
    a_cleaned[a>2] = 0
    a_cleaned[a==2] = -1

    # fill history matrices with n_back information
    for nn in range(n_back):
        a_hist[(1+nn):,nn] = a_cleaned[0:num_trials-nn-1]
        R_hist[(1+nn):,nn] = R[0:num_trials-nn-1]       
        uR_hist[(1+nn):,nn] = uR[0:num_trials-nn-1]  
    
    return np.fliplr(a_hist), np.fliplr(R_hist), np.fliplr(uR_hist)

def PA_logit(a,p_logit):
    PA = (np.nansum((a==1) & (p_logit>=0.5)) + np.nansum((a==2) & (p_logit<0.5)))/sum((a==1) | (a==2))
    return PA

# Accuracy and loglik from general logit function
def PA_LL_logit(a,p_logit):
    PA = (np.nansum((a==1) & (p_logit>=0.5)) + np.nansum((a==2) & (p_logit<=0.5)))/sum((a==1) | (a==2))
    loglik = (np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    return PA, loglik


# Basic logistic Regression fitting functions
def logistic_prepare_predictors(a,R,n_back,mdl_type='RUC'):
    num_trials = len(a)
    
    # Get history matrices
    a_hist, R_hist,_ = prepare_hist_matrix(a,R,n_back)

    # create and align component arrays
    past_rc = np.multiply(a_hist,R_hist)
    past_uc = np.multiply(a_hist,R_hist==0)
    past_c = a_hist
    
    # identifty choice trials
    choice_trials = (a==1)|(a==2)
    left_trials = (a==1)+0
    left_trials[a==2]=-1
    left_choice_trials = left_trials[choice_trials]

    # create output arrays: predictors and targets
    glm_target = left_choice_trials
    if mdl_type=='RUC':
        glm_mat = np.concatenate((past_rc[choice_trials,:], \
                                  past_uc[choice_trials,:], \
                                  past_c[choice_trials,:]),axis=1)
    elif mdl_type=='RU':
        glm_mat = np.concatenate((past_rc[choice_trials,:], \
                                  past_uc[choice_trials,:]),axis=1)
    elif mdl_type=='RC':
        glm_mat = np.concatenate((past_rc[choice_trials,:], \
                                  past_c[choice_trials,:]),axis=1)
    elif mdl_type=='R':
        glm_mat = (past_rc[choice_trials,:])
    
    return glm_mat, glm_target

def logistic_calc_loglik(glm_mat,glm_target,coef,intercept,n_back):
    # create the matrix of coefficients for easy elementwise multiplication
    coef_mat = np.tile(np.reshape(coef,[len(coef),1]),len(glm_mat))
    coef_mat = coef_mat.transpose()
    sum_logit = np.sum(glm_mat*coef_mat,axis=1)+intercept

    # convert the sum into a probability, and prediction
    p_logit = 1/(1+np.exp(-sum_logit))
    predicted_temp = np.round(p_logit)
    loglik = ((sum(np.log(p_logit[glm_target==1])) + 
               sum(np.log(1 - p_logit[glm_target==-1]))))
    
    n_params = (coef!=0).sum()+1
    aic = 2*n_params - 2*loglik
    return loglik, aic, p_logit

# Optogenetics logistic regression
def prepare_opto_predictor_df(a,R,Opto):
    num_trials = len(a)
    n_back = 15
    
    # Get history matrices
    a_hist, R_hist,uR_hist = prepare_hist_matrix(a,R,n_back)

    # create and align component arrays
    past_rc = np.multiply(a_hist,R_hist)
    past_uc = np.multiply(a_hist,R_hist==0)
    past_c = a_hist
    past_R = R_hist
    past_uR = uR_hist
    
    # Codify the dependent variable: choice
    dv = (a==1).astype(int)
    dv = dv.reshape([len(dv),1])
    # codify constant
    const = np.ones(len(a)).reshape([len(a),1])
    
    opto_on = np.where(Opto)[0]
    choice_trials = (a==1)|(a==2)
    opto_choice_trials = ((a==1) | (a==2)) & Opto
    ctrl_choice_trials = ((a==1) | (a==2)) & (Opto==0)
    opto_inds = np.where(opto_choice_trials)[0]
    
    # Make the control trial inds
    opto_shift_1 = opto_on+1; opto_shift_1 = opto_shift_1[opto_shift_1<len(a)]
    opto_shift_2 = opto_on+2; opto_shift_2 = opto_shift_2[opto_shift_2<len(a)]
    opto_shift_3 = opto_on+3; opto_shift_3 = opto_shift_3[opto_shift_3<len(a)]
    post3_opto = np.ones(len(Opto)).astype(int)
    post3_opto[opto_on] = 0
    post3_opto[opto_shift_1] = 0
    post3_opto[opto_shift_2] = 0
    post3_opto[opto_shift_3] = 0
    post3_opto[a>2] = 0
    ctrl_inds = np.where(post3_opto)[0]
    # ctrl_inds = np.where(ctrl_choice_trials)[0]
    
    post_opto_choice_trials = np.zeros(len(Opto)).astype(int) 
    post_opto_inds = opto_shift_1[choice_trials[opto_shift_1]==1]
    post_opto_choice_trials[post_opto_inds] = 1
    
    opto_choice_trials = opto_choice_trials.reshape([len(opto_choice_trials),1])
    post_opto_choice_trials = post_opto_choice_trials.reshape([len(post_opto_choice_trials),1])
    trials_to_use = np.zeros(len(a))
    trials_to_use[opto_inds]=1
    trials_to_use[ctrl_inds]=1
    trials_to_use[post_opto_inds]=1
    trials_to_use = trials_to_use.astype(bool)
    
    temp_df = pd.DataFrame(np.hstack([dv[trials_to_use], 
                                      opto_choice_trials[trials_to_use],
                                      post_opto_choice_trials[trials_to_use],
                                      const[trials_to_use],
                                      past_R[trials_to_use,:],
                                      past_uR[trials_to_use,:],
                                      past_rc[trials_to_use,:],
                                      past_uc[trials_to_use,:],
                                      past_c[trials_to_use,:]]),
                  columns=['DV','opto','post_opto','const',
                           'Rp15','Rp14','Rp13','Rp12','Rp11','Rp10','Rp9','Rp8','Rp7','Rp6','Rp5','Rp4','Rp3','Rp2','Rp1',
                           'uRp15','uRp14','uRp13','uRp12','uRp11','uRp10','uRp9','uRp8','uRp7','uRp6','uRp5','uRp4','uRp3','uRp2','uRp1',
                           'RCp15','RCp14','RCp13','RCp12','RCp11','RCp10','RCp9','RCp8','RCp7','RCp6','RCp5','RCp4','RCp3','RCp2','RCp1',
                           'UCp15','UCp14','UCp13','UCp12','UCp11','UCp10','UCp9','UCp8','UCp7','UCp6','UCp5','UCp4','UCp3','UCp2','UCp1',
                           'Cp15','Cp14','Cp13','Cp12','Cp11','Cp10','Cp9','Cp8','Cp7','Cp6','Cp5','Cp4','Cp3','Cp2','Cp1'])

    return temp_df  

def prepare_opto_glm_5(predictor_df, mdl_type):
    n_back = 5
    # Prepare matrices
    opto_params = predictor_df[['opto']].to_numpy()
    glm_target = predictor_df['DV'].values
    
    
    if mdl_type == 'RUC':
        history_params = predictor_df[['RCp5','RCp4','RCp3','RCp2','RCp1',
                                       'UCp5','UCp4','UCp3','UCp2','UCp1',
                                       'Cp5','Cp4','Cp3','Cp2','Cp1']].to_numpy()
        glm_mat = np.concatenate((history_params*np.tile(opto_params==0,[1,3*n_back]),
                                  history_params*np.tile(opto_params==1,[1,3*n_back]),
                                  opto_params
                                 ),axis=1)
    elif mdl_type == 'RU':
        history_params = predictor_df[['RCp5','RCp4','RCp3','RCp2','RCp1',
                                       'UCp5','UCp4','UCp3','UCp2','UCp1']].to_numpy()
        glm_mat = np.concatenate((history_params*np.tile(opto_params==0,[1,2*n_back]),
                                  history_params*np.tile(opto_params==1,[1,2*n_back]),
                                  opto_params
                                 ),axis=1)
    elif mdl_type == 'RC':
        history_params = predictor_df[['RCp5','RCp4','RCp3','RCp2','RCp1',
                                       'Cp5','Cp4','Cp3','Cp2','Cp1']].to_numpy()
        glm_mat = np.concatenate((history_params*np.tile(opto_params==0,[1,2*n_back]),
                                  history_params*np.tile(opto_params==1,[1,2*n_back]),
                                  opto_params
                                 ),axis=1)
    elif mdl_type == 'R':
        history_params = predictor_df[['RCp5','RCp4','RCp3','RCp2','RCp1']].to_numpy()
        glm_mat = np.concatenate((history_params*np.tile(opto_params==0,[1,1*n_back]),
                                  history_params*np.tile(opto_params==1,[1,1*n_back]),
                                  opto_params
                                 ),axis=1)
        
    return glm_mat, glm_target, opto_params, opto_params

def logistic_calc_loglik_pa(glm_mat, glm_target, mdl_coef):
    # For the full session, full model
    # Note the different input structure to the other logistic loglik function
    opto_params = glm_mat[:,-1]   
    intercept = mdl_coef[-1]
    coef = mdl_coef[:-1]
    coef[-1] = coef[-1] - intercept
    
    # create the matrix of coefficients for easy elementwise multiplication
    coef_mat = np.tile(np.reshape(coef,[len(coef),1]),len(glm_mat))
    coef_mat = coef_mat.transpose()
    sum_logit = np.sum(glm_mat*coef_mat,axis=1)+intercept

    # convert the sum into a probability, and prediction
    p_logit = 1/(1+np.exp(-sum_logit))
    predicted_temp = np.round(p_logit)
    loglik = ((sum(np.log(p_logit[glm_target==1])) + 
               sum(np.log(1 - p_logit[glm_target==0]))))
    
    pa = (sum((predicted_temp==1) & (glm_target==1)) + 
          sum((predicted_temp==0) & (glm_target==0)) ) / len(glm_target)
    # n_params = (coef!=0).sum()+1
    # aic = 2*n_params - 2*loglik
    return loglik/len(glm_target), pa

def logistic_opto_calc_loglik(glm_mat, glm_target, mdl_coef):
    # For control and opto trials seperately
    opto_params = glm_mat[:,-1]   
    n_opto = sum(opto_params)
    n_ctrl = sum(opto_params==0)
    intercept = mdl_coef[-1]
    coef = mdl_coef[:-1]
    coef[-1] = coef[-1] - intercept
    
    # create the matrix of coefficients for easy elementwise multiplication
    coef_mat = np.tile(np.reshape(coef,[len(coef),1]),len(glm_mat))
    coef_mat = coef_mat.transpose()
    sum_logit = np.sum(glm_mat*coef_mat,axis=1)+intercept

    # convert the sum into a probability, and prediction
    p_logit = 1/(1+np.exp(-sum_logit))
    predicted_temp = np.round(p_logit)
    
    opto_p_logit = p_logit[opto_params==1]
    ctrl_p_logit = p_logit[opto_params==0]
    opto_predicted = predicted_temp[opto_params==1]
    ctrl_predicted  = predicted_temp[opto_params==0]
    opto_glm_target = glm_target[opto_params==1]
    ctrl_glm_target = glm_target[opto_params==0]
           
    opto_loglik = ((sum(np.log(opto_p_logit[opto_glm_target==1])) +
                    sum(np.log(1 - opto_p_logit[opto_glm_target==0]))))
    ctrl_loglik = ((sum(np.log(ctrl_p_logit[ctrl_glm_target==1])) +
                    sum(np.log(1 - ctrl_p_logit[ctrl_glm_target==0]))))
    opto_pa = (sum((opto_predicted==1) & (opto_glm_target==1)) + 
               sum((opto_predicted==0) & (opto_glm_target==0)) ) / n_opto
    ctrl_pa = (sum((ctrl_predicted==1) & (ctrl_glm_target==1)) + 
               sum((ctrl_predicted==0) & (ctrl_glm_target==0)) ) / n_ctrl
    return opto_loglik/n_opto, ctrl_loglik/n_ctrl, opto_pa, ctrl_pa

def prepare_opto_behlogit_5(predictor_df):
    n_back = 5
    # Prepare matrices
    opto_params = predictor_df[['opto']].to_numpy()
    glm_target = predictor_df['DV'].values
    
    R_hist = history_params = predictor_df[['Rp5','Rp4','Rp3','Rp2','Rp1']].to_numpy()
    uR_hist = history_params = predictor_df[['uRp5','uRp4','uRp3','uRp2','uRp1']].to_numpy()
    a_hist = history_params = predictor_df[['Cp5','Cp4','Cp3','Cp2','Cp1']].to_numpy()
        
    return a_hist, R_hist, uR_hist, glm_target, opto_params

def prepare_opto_behlogit_10(predictor_df):
    n_back = 10
    # Prepare matrices
    opto_params = predictor_df[['opto']].to_numpy()
    glm_target = predictor_df['DV'].values
    R_hist = history_params = predictor_df[['Rp10','Rp9','Rp8','Rp7','Rp6','Rp5',
                                            'Rp4','Rp3','Rp2','Rp1']].to_numpy()
    uR_hist = history_params = predictor_df[['uRp10','uRp9','uRp8','uRp7','uRp6',
                                             'uRp5','uRp4','uRp3','uRp2','uRp1']].to_numpy()
    a_hist = history_params = predictor_df[['Cp10','Cp9','Cp8','Cp7','Cp6','Cp5',
                                            'Cp4','Cp3','Cp2','Cp1']].to_numpy()

    return a_hist, R_hist, uR_hist, glm_target, opto_params

def prepare_opto_behlogit_15(predictor_df):
    n_back = 15
    # Prepare matrices
    opto_params = predictor_df[['opto']].to_numpy()
    glm_target = predictor_df['DV'].values
    R_hist = history_params = predictor_df[['Rp15','Rp14','Rp13','Rp12','Rp11','Rp10','Rp9',
                                            'Rp8','Rp7','Rp6','Rp5','Rp4','Rp3','Rp2','Rp1']].to_numpy()
    uR_hist = history_params = predictor_df[['uRp15','uRp14','uRp13','uRp12','uRp11','uRp10',
                                             'uRp9','uRp8','uRp7','uRp6','uRp5','uRp4','uRp3','uRp2','uRp1']].to_numpy()
    a_hist = history_params = predictor_df[['Cp15','Cp14','Cp13','Cp12','Cp11','Cp10','Cp9',
                                            'Cp8','Cp7','Cp6','Cp5','Cp4','Cp3','Cp2','Cp1']].to_numpy()

    return a_hist, R_hist, uR_hist, glm_target, opto_params

# Behavior model functions
### Behavior logits with different exponential assumptions
def logit_exp_r(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    const = params[2]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc) 

    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_exp_ru(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc) 
        
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_exp_rc(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c = params[2]
    tau_c = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_exp_ruc(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    const = params[6]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc))
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik

### Behavior logits with different hyperbolic assumptions
def logit_hyp_r(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    const = params[2]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc) 

    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))

    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_hyp_ru(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_hyp_rc(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c = params[2]
    tau_c = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_c = np.flip(1/(1+np.arange(0,n_back)/tau_c))
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_hyp_ruc(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    const = params[6]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
        
    g_c = np.flip(1/(1+np.arange(0,n_back)/tau_c))
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc))
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
### Behavior logits with different combination of assumptions
def logit_hyp_r_exp_u(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_exp_r_hyp_u(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_hyp_r_exp_c(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c = params[2]
    tau_c = np.exp(params[3])
    const = params[4]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
def logit_hyp_r_exp_uc(params,a,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    const = params[6]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
        
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
        
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)

    # convert the sum into a probability, calculate negative log likelihood
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc))
                   + beta_c*(np.dot(a_hist,g_c)) )
    p_logit = 1/(1+np.exp(-sum_weights))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    # If this is for fitting, return only nloglik
    if fit_bool:
        return nloglik
    else:
        return p_logit, nloglik
    
# Quasi-hyperbolic
def logit_quasi_fit(params,tau_vec,a,a_hist,R_hist,fit_bool):
    n_back = np.shape(a_hist)[1]
    const = params[0]
    filter_weight = params[1:]
    
    # Prepare non-parametric filter
    n_tau = len(tau_vec)
    if len(tau_vec)!=len(filter_weight):
        print('mismatch in number of taus and provided weights')
    x_ind = np.flip(-np.arange(0,n_back))
    x_mat = np.tile(x_ind,[n_tau,1])
    tau_mat = np.tile(tau_vec,[n_back,1]).transpose()
    beta_mat = np.tile(filter_weight,[n_back,1]).transpose()
    exp_mat = beta_mat*np.exp(x_mat/tau_mat)
    np_filter = np.mean(exp_mat,axis=0)
    
    # Convolve filter to behavior matrix
    past_r = np.multiply(a_hist,R_hist)
    gWR = np_filter
    gWR[np.isnan(gWR)]=0. # Catch to remove nans
    sum_logit = const + (np.dot(past_r,gWR))
    p_logit = 1/(1+np.exp(-sum_logit))
    nloglik = -(np.nansum(np.log(p_logit[a==1]))+np.nansum(np.log(1-p_logit[a==2])))
    
    if fit_bool:
        return nloglik
    else:
        return p_logit, np_filter, const, nloglik

# Cell model fit estimation
def calc_mdl_loglik(sse, n_obs):
    loglik = -n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
    return loglik, loglik/n_obs
def calc_aic_sse(sse, n_obs, n_params):
    return n_obs*np.log(sse/n_obs) + 2*n_params
def calc_aic_loglik(loglik, n_params):
    return 2*n_params - 2*loglik 

## Exponential cell model functions
def exp_filter(a_hist,R_hist,tau,hist_type):
    n_back = np.shape(a_hist)[1]    
    if hist_type=='r':
        past = np.multiply(a_hist,R_hist)
    elif hist_type=='ur':
        past = np.multiply(a_hist,R_hist==0)
    elif hist_type=='c':
        past = a_hist
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc) 
    # convert the components into a sum
    sum_weights = (np.dot(past,g_rc))
    return sum_weights

def reg_null_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_c0 = params[0]
    const = params[1]

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + (np.dot(c0,beta_c0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_Rp1_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    beta_c0 = params[1]
    const = params[2]
    
    past_r = np.multiply(a_hist,R_hist)[:,-1]
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + (np.dot(past_r,beta_rc))+ (np.dot(c0,beta_c0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr

def reg_exp_r_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c0 = params[2]
    const = params[3]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + beta_rc*(np.dot(past_r,g_rc)) + (np.dot(c0,beta_c0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_ru_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_rc_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c = params[2]
    tau_c = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_c*(np.dot(a_hist,g_c)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr

def reg_exp_ruc_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    beta_c0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + beta_c*(np.dot(a_hist,g_c))
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr    

# constrained functions   
def reg_exp_ru_con_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = np.sign(beta_rc)*params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)     
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
        
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
     
def reg_exp_ruc_con_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = np.sign(beta_rc)*params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    beta_c0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)     
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    g_c = np.flip(np.exp(-np.arange(0,n_back)/tau_c)) 
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + beta_c*(np.dot(a_hist,g_c))
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
## Hyperbolic cell functions
def hyp_filter(a_hist,R_hist,tau,hist_type):
    n_back = np.shape(a_hist)[1]    
    if hist_type=='r':
        past = np.multiply(a_hist,R_hist)
    elif hist_type=='ur':
        past = np.multiply(a_hist,R_hist==0)
    elif hist_type=='c':
        past = a_hist
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum
    sum_weights = (np.dot(past,g_rc))
    return sum_weights

def reg_hyp_r_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c0 = params[2]
    const = params[3]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + beta_rc*(np.dot(past_r,g_rc)) + (np.dot(c0,beta_c0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik 
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_ru_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_rc_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_c = params[2]
    tau_c = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    g_c = np.flip(1/(1+np.arange(0,n_back)/tau_c))
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_c*(np.dot(a_hist,g_c)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr

def reg_hyp_ruc_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    beta_c0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    g_c = np.flip(1/(1+np.arange(0,n_back)/tau_c))
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + beta_c*(np.dot(a_hist,g_c))
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik 
    else:
        return sum_weights, sse, rsq, snr    

# constrained function    
def reg_hyp_ru_con_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = np.sign(beta_rc)*params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)     
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
        
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik 
    else:
        return sum_weights, sse, rsq, snr
        
def reg_hyp_ruc_con_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = np.sign(beta_rc)*params[2]
    tau_uc = np.exp(params[3])
    beta_c = params[4]
    tau_c = np.exp(params[5])
    beta_c0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)
    g_c = np.flip(1/(1+np.arange(0,n_back)/tau_c))
    g_c[np.isnan(g_c)]=0.  # Catch to remove nans
    if sum(g_c)!=0:
        g_c = g_c/sum(g_c)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + beta_c*(np.dot(a_hist,g_c))
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_r_hyp_u_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_r_exp_u_c0(params,y,c0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_c0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)  
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(c0,beta_c0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
    
# Choice-aligned cell analysis (Early-ITI)
def reg_null_early(params,y,c0,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc0 = params[0]
    beta_uc0 = params[1]
    beta_r0 = params[2]
    const = params[3]

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + (np.dot(c0*r0,beta_rc0)) + (np.dot(c0*(r0==0),beta_uc0)) +
                   (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_r_early(params,y,c0,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_rc0 = params[2]
    beta_uc0 = params[3]
    beta_r0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + 
                   (np.dot(c0*r0,beta_rc0)) + (np.dot(c0*(r0==0),beta_uc0)) + (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_ru_early(params,y,c0,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_rc0 = params[4]
    beta_uc0 = params[5]
    beta_r0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) + 
                   (np.dot(c0*r0,beta_rc0)) + (np.dot(c0*(r0==0),beta_uc0)) + (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_r_early(params,y,c0,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_rc0 = params[2]
    beta_uc0 = params[3]
    beta_r0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + 
                   (np.dot(c0*r0,beta_rc0)) + (np.dot(c0*(r0==0),beta_uc0)) + (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik 
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_ru_early(params,y,c0,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_rc0 = params[4]
    beta_uc0 = params[5]
    beta_r0 = params[6]
    const = params[7]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) + beta_uc*(np.dot(past_ur,g_uc)) + 
                   (np.dot(c0*r0,beta_rc0)) + (np.dot(c0*(r0==0),beta_uc0)) + (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr

# Choice-aligned cell analysis (Late-ITI)
def reg_null_late(params,y,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_r0 = params[0]
    const = params[1]

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + (np.dot(r0,beta_r0)) )
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_r_late(params,y,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_r0 = params[2]
    const = params[3]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + beta_rc*(np.dot(past_r,g_rc)) + (np.dot(r0,beta_r0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_exp_ru_late(params,y,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_r0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(np.exp(-np.arange(0,n_back)/tau_rc)) 
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(np.exp(-np.arange(0,n_back)/tau_uc)) 
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(r0,beta_r0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_r_late(params,y,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_r0 = params[2]
    const = params[3]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)
    
    # convert the components into a sum, calculate residual sum of squares
    sum_weights = const + beta_rc*(np.dot(past_r,g_rc)) + (np.dot(r0,beta_r0))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik 
    else:
        return sum_weights, sse, rsq, snr
    
def reg_hyp_ru_late(params,y,r0,a_hist,R_hist,uR_hist,fit_bool):
    n_back = np.shape(a_hist)[1]    
    beta_rc = params[0]
    tau_rc = np.exp(params[1])
    beta_uc = params[2]
    tau_uc = np.exp(params[3])
    beta_r0 = params[4]
    const = params[5]
    
    past_r = np.multiply(a_hist,R_hist) # already flipped
    past_ur = np.multiply(a_hist,uR_hist)
    g_rc = np.flip(1/(1+np.arange(0,n_back)/tau_rc))
    g_rc[np.isnan(g_rc)]=0.  # Catch to remove nans
    if sum(g_rc)!=0:
        g_rc = g_rc/sum(g_rc)    
    g_uc = np.flip(1/(1+np.arange(0,n_back)/tau_uc))
    g_uc[np.isnan(g_uc)]=0.  # Catch to remove nans
    if sum(g_uc)!=0:
        g_uc = g_uc/sum(g_uc)

    # convert the components into a sum, calculate residual sum of squares
    sum_weights = (const + beta_rc*(np.dot(past_r,g_rc)) 
                   + beta_uc*(np.dot(past_ur,g_uc)) 
                   + (np.dot(r0,beta_r0)))
    residual = sum_weights - y
    sse = np.sum(np.square(residual))
    tss = sum((y-np.mean(y))**2)
    rsq = 1-sse/tss
    snr = np.std(sum_weights)**2/np.std(residual)**2
    
    # If this is for fitting, return only sse
    if fit_bool==1:
        return sse 
    elif fit_bool==2:
        n_obs = len(y)
        nloglik = n_obs/2*(np.log(2*np.pi*sse) - np.log(n_obs) + 1)
        return nloglik
    else:
        return sum_weights, sse, rsq, snr