# import modpods and other libraries
import sys
sys.path.append("G:/My Drive/modpods")
import modpods
#print(modpods)
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import dill as pickle
import math
import datetime
import time

# this script is going to look almost identical to CAMELS_modpods_eval.py

data_folder_path = "G:/My Drive/PhD Admin and Notes/paper1/revisions-code/glwa_data/"

# all the GLWA data has the same start and end dates
# timestep is hourly. so it might be helpful to use shifting
# just train on one summer and test on the previous. don't want to deal with snowmelt dominated events
windup_timesteps = 2*24*30 # two months
train_start = datetime.date(month=5,day=1,year=2022) # six months of training
train_end = datetime.date(month=11,day=1,year=2022)
eval_start = datetime.date(month=5,day=1,year=2021)
eval_end = datetime.date(month=11,day=1,year=2021)


# model parameters
max_polyorder = 3
max_iter = 100
max_transforms = 2
for subdir, dirs, files in os.walk(data_folder_path):
    print(subdir)
    for file in files:
        # train and store results for shifted and not shifted
        print(file)
        # NO SHIFT TRAINING then SHIFT TRAINING
        shift = False
        for i in range(0,2): # only do shifted training
            if (i < 1): # first time, haven't loaded data yet
                print("no shift")
                df = pd.read_csv(str(os.path.join(subdir, file)))
                print(df)

                
                df.set_index('DateTime', inplace=True)
                df.index = pd.DatetimeIndex(df.index)

                #df = df.tz_localize("US/Eastern", ambiguous = 'NaT',nonexistent='NaT')
                df = df[~df.index.duplicated(keep='first')]

                rain_gage_columns = df.columns[-4:-1]
                realtime_flow_col_idx = np.argwhere(df.columns.str.contains('Realtime-f'))[0][0]

                print(realtime_flow_col_idx)

                print(rain_gage_columns)
                df['Discharge [CFS]'] = df[df.columns[realtime_flow_col_idx]]  #- df[df.columns[-1]] # subtract "greenline" which is the dry weather flow estimate
                # make the df['Discharge [CFS]'] absolute value
                #df['Discharge [CFS]'] = df['Discharge [CFS]'].abs()

                #df['Discharge [CFS]'] = df[df.columns[realtime_flow_col_idx]] # don't subtract dry weather flow estimate
                og = df[df.columns[realtime_flow_col_idx]]# - df[df.columns[-2]] # subtract greenline
                
                df = pd.concat((df[rain_gage_columns],df['Discharge [CFS]']), axis='columns' )# get rid of everything except rain gages and discharge
                df[rain_gage_columns].fillna(0) # fill missing values with 0
                df['Discharge [CFS]'].interpolate(method='time',inplace=True,limit_area='inside') # interpolate missing values in discharge
                print(df)
                # smooth discharge cfs using a X hour rolling mean (closed left to prevent anticipatory response)
                df['Discharge [CFS]'] = df['Discharge [CFS]'].rolling(window=6,closed='left',center=False).mean()
                df.dropna(inplace=True) # drop any rows with nan values still (should just be on outsides)
                
                print(df)
                # assert there are no NaN values in df
                assert df.isnull().values.any() == False
                print(df.isnull().values.any())
                '''
                df['Discharge [CFS]'][:500].plot(alpha=0.5)
                og[:500].plot(alpha=0.5)
                plt.show()
                '''
                if(shift): # not training the unshifted version
                    continue


            else:
                print("shift is on")
                # shift the forcing back one timestep (one hour) to make the system causal
                #print(df[['OBS_RUN','RAIM']])
                #print(df[rain_gage_columns])
                df[rain_gage_columns] = df[rain_gage_columns].shift(-1)
                df.dropna(inplace=True,subset=rain_gage_columns) # drop that last column where the rain gages are nan now
                #print(df[['OBS_RUN','RAIM']])
                shift=True
                print(df)

            

            df_train = df.loc[train_start-datetime.timedelta(hours=windup_timesteps):train_end+datetime.timedelta(days=1),:] 
            print("training data")
            print(df_train)
            # make df_eval a subset of the rows of df using eval_start and eval_end 
            df_eval = df.loc[eval_start-datetime.timedelta(hours=windup_timesteps):eval_end+datetime.timedelta(days=1),:]  # data for evaluation, not used in training
            print("evaluation data")
            print(df_eval)
            for polyorder in range(1, max_polyorder+1):

                start = time.perf_counter()
                rainfall_runoff_model = modpods.delay_io_train(df_train,['Discharge [CFS]'],rain_gage_columns,windup_timesteps=windup_timesteps,
                init_transforms=1, max_transforms=max_transforms,max_iter=max_iter,
                poly_order=polyorder, verbose=False, bibo_stable=True)
                end = time.perf_counter()
                training_time_minutes = (end-start)/60
                            



                if (shift):
                    results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/glwa_modpods_results/" + str(file)[:-4] + '_po_'+  str(polyorder) + '_shift/')
                else:
                    results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/glwa_modpods_results/" + str(file)[:-4] +'_po_'+ str(polyorder) + '_no_shift/')
                if not os.path.exists(results_folder_path):
                    # Create a new directory because it does not exist
                    os.makedirs(results_folder_path)
                with open(str(results_folder_path  + 'rainfall_runoff_model'),'wb') as f:
                    pickle.dump(rainfall_runoff_model,f) # binary of full training results
                perf = pd.DataFrame.from_dict(rainfall_runoff_model[1]['final_model']['error_metrics'],orient='columns',dtype='float')
                for num_transforms in range(2, max_transforms+1):
                    perf = pd.concat([perf, pd.DataFrame.from_dict(rainfall_runoff_model[num_transforms]['final_model']['error_metrics'])])
                perf.index = range(1, max_transforms+1)
                perf['training_time_minutes'] = training_time_minutes
                perf.to_csv(str(results_folder_path +'training_error_metrics.csv'))
                print(perf)
                del perf
                # plot the results
                fig, ax = plt.subplots(max_transforms,1,figsize=(10,10))
                for i in range(0,max_transforms):
                    ax[i].plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[i+1]['final_model']['response']['Discharge [CFS]'][windup_timesteps+1:],label='observed')
                    if (not rainfall_runoff_model[i+1]['final_model']['diverged']): # simulation didn't diverge, so the simulated data is valid
                        ax[i].plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[i+1]['final_model']['simulated'][:,0],label='simulated')
                    ax[i].set_title(str(str(i+1) + ' transformation(s)'))
                    if (i<1):
                        ax[i].legend()
                fig.suptitle("training")
                plt.tight_layout()
                plt.savefig(str(results_folder_path + "training_viz.png"), dpi=300,bbox_inches='tight')
                plt.savefig(str(results_folder_path + "training_viz.svg"), dpi=300,bbox_inches='tight')
                plt.close()
                diverged_sims = list()
                eval_simulations = list() # for plotting
                for num_transforms in range(1, max_transforms+1):
                    eval_sim = modpods.delay_io_predict(rainfall_runoff_model, df_eval, num_transforms,evaluation=True)
                    eval_simulations.append(eval_sim['prediction'])
                    diverged_sims.append(eval_sim['diverged'])
                    if (num_transforms < 2):
                        eval_perf = pd.DataFrame.from_dict(eval_sim['error_metrics'],orient='columns',dtype='float')
                        print(eval_perf)
                    else:
                        eval_perf = pd.concat([eval_perf, pd.DataFrame.from_dict(eval_sim['error_metrics'])])
                        print(eval_perf)
                #print(eval_perf)
                eval_perf.index = range(1, max_transforms+1)
                eval_perf['training_time_minutes'] = training_time_minutes
                eval_perf.to_csv(str(results_folder_path +'eval_error_metrics.csv'))
                print(eval_perf)
                del eval_perf
                # plot the results
                fig, ax = plt.subplots(max_transforms,1,figsize=(10,10))
                for i in range(0,max_transforms):
                    ax[i].plot(df_eval.index[windup_timesteps+1:],df_eval['Discharge [CFS]'][windup_timesteps+1:],label='observed')
                    if (not diverged_sims[i]):
                        ax[i].plot(df_eval.index[windup_timesteps+1:],eval_simulations[i],label='simulated')
                    ax[i].set_title(str(str(i+1) + ' transformation(s)'))
                    if (i<1):
                        ax[i].legend()
                fig.suptitle("evaluation")
                plt.tight_layout()

                plt.savefig(str(results_folder_path + "eval_viz.png"), dpi=300,bbox_inches='tight')
                plt.savefig(str(results_folder_path + "eval_viz.svg"), dpi=300,bbox_inches='tight')
                plt.close()


             