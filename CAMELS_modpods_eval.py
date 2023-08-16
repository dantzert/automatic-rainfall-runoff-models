# import modpods and other libraries
from re import T
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


# find all basins modeled by all benchmarks (only need to run this once)
'''
print("Loaded data from pre-computed pickle file")
with open("all_metrics.p", "rb") as fp:
    all_metrics = pickle.load(fp)

basins = frozenset(list(all_metrics["NSE"]["ealstm_NSE"]["ensemble"].keys()))
for model, results in all_metrics["NSE"]["benchmarks"].items():
    basins = basins.intersection(list(results.keys()))
len(basins)
print(type(basins))
print(basins)
# write the set basins to a text file 
with open("benchmark_basins.txt", "w") as fp:
    for basin in basins:
            fp.write(basin + ",")
'''



# load in the text file with the site ids of the sites we want to train (447 of them)
# load benchmark_basins.txt and save it as a set
with open("benchmark_basins.txt", "r") as fp:
    basins = fp.read()
    basins = basins.split(",")
    basins = frozenset(basins)
    #print(type(basins))
    #print(len(basins)) # 448


# folder_path = "G:/My Drive/PhD Admin and Notes/paper1/CAMELS/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output/flow_timeseries/daymet/"
# "G:/My Drive/PhD Admin and Notes/paper1/CAMELS/model_output_maurer/model_output/flow_timeseries/maurer/02/01333000_05_model_output.txt"
data_folder_path = "G:/My Drive/PhD Admin and Notes/paper1/CAMELS/model_output_maurer/model_output/flow_timeseries/maurer/"
bbasins = set()
for subdir, dirs, files in os.walk(data_folder_path):
    #print(subdir)
    #print('\n')
    for file in files:
        if ("05_model_output.txt" in str(os.path.join(subdir,file) ) ):
            #print(str(file)[0:8])
            if (str(file)[0:8] in basins): # consistency check
                bbasins.add(str(file)[0:8])
            #print(str(os.path.join(subdir, file)))
'''
print("total files")
print(len(bbasins)) # 441
print("difference")
print(basins.difference(bbasins))
# this is fine for now.
'''

# the training
# ref for training setup: https://hess.copernicus.org/articles/23/5089/2019/ sections 2.4-2.6
# training set up parameters
windup_timesteps = 269 # days of windup, per kratzert 2018
train_start = datetime.date(month=10,day=1,year=1999)
train_end = datetime.date(month=9,day=30,year=2008)
eval_start = datetime.date(month=10,day=1,year=1989)
eval_end = datetime.date(month=9,day=30,year=1999)
'''
# shorter times for debugging
train_start = datetime.date(month=1,day=1,year=1999)
train_end = datetime.date(month=1,day=1,year=2000)
eval_start = datetime.date(month=1,day=1,year=2000)
eval_end = datetime.date(month=1,day=1,year=2001)
'''
# model parameters
max_polyorder = 3
max_iter = 100
max_transforms = 2
for subdir, dirs, files in os.walk(data_folder_path):
    print(str(subdir))
    print(str(subdir)[-2:])
    try:
        int(str(subdir)[-2:]) # if this fails, it's not one of the huc2 region subfolders
    except:
        continue
    #print('\n')
    if (int(str(subdir)[-2:]) > 0): # fo training subsets of the data
        print(subdir)
        for file in files:
            if ("05_model_output.txt" in str(os.path.join(subdir,file) ) ):
                if (str(file)[0:8] in bbasins): # in the set that has benchmark results
                    site_id = str(file)[0:8]
                    # train and store results for shifted and not shifted
                    # store results for using the first/last 15 years as training/evaluation with "eval_first" or "eval_last"
                    # filename as [usgs site id]_["shifted" or "noshift"]_["training" or "eval"]_["plot" or "performance"].['png', 'svg', 'csv']
                    # also save the trained models as binaries (pickle)

                    # NO SHIFT TRAINING then SHIFT TRAINING
                    # still need to implement front vs back training. might not be necessary even though it would be more complete that way
                    shift = True#False
                    for i in range(0,2): # only do shifted training
                        if (i < 1): # first time, haven't loaded data yet
                            print("no shift")
                            df = pd.read_csv(str(os.path.join(subdir, file)), sep='\s+')
                            print(df)
                            # combine the columns YR, MNTH, DY, and YR into a single datetime column
                            df.rename({'YR':'year','MNTH':'month','DY':'day','HR':'hour'},axis=1,inplace=True)
                            df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])

                            # set the index to the datetime column
                            df.set_index('datetime',inplace=True)

                            # subtract the minmum value of OBS_RUN from OBS_RUN (remove constant offset)
                            print("constant offset in OBS_RUN (minimum of timeseries):", df.OBS_RUN.min())
                            df.OBS_RUN = df.OBS_RUN - df.OBS_RUN.min()
                            print(df)
                            if(shift):
                                continue


                        else:
                            print("shift is on")
                            # shift the forcing back one timestep (one day) to make the system causal
                            #print(df[['OBS_RUN','RAIM']])
                            df.RAIM = df.RAIM.shift(-1)
                            df.dropna(inplace=True)
                            #print(df[['OBS_RUN','RAIM']])
                            shift=True

                        df_train = df.loc[train_start-datetime.timedelta(days=windup_timesteps):train_end+datetime.timedelta(days=1),:] 
                        #print(df_train)
                        df_eval = df.loc[eval_start-datetime.timedelta(days=windup_timesteps):eval_end+datetime.timedelta(days=1),:]  # data for evaluation, not used in training
                        #print(df_eval)
                        for polyorder in range(1, max_polyorder+1):
                            start = time.perf_counter()
                            rainfall_runoff_model = modpods.delay_io_train(df_train, ['OBS_RUN'],['RAIM'],windup_timesteps=windup_timesteps,
                            init_transforms=1, max_transforms=max_transforms,max_iter=max_iter,
                            poly_order=polyorder, verbose=False, bibo_stable=True)
                            end = time.perf_counter()
                            training_time_minutes = (end-start)/60
                            
                            if (shift):
                                results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/" + str(site_id) + '_po_'+  str(polyorder) + '_shift/')
                            else:
                                results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/" + str(site_id) +'_po_'+ str(polyorder) + '_no_shift/')
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
                                ax[i].plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[i+1]['final_model']['response']['OBS_RUN'][windup_timesteps+1:],label='observed')
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
                                ax[i].plot(df_eval.index[windup_timesteps+1:],df_eval['OBS_RUN'][windup_timesteps+1:],label='observed')
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


             




 





                


