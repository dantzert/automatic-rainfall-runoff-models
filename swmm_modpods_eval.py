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
import math
#!pip3 install pyswmm
import pyswmm 
#!pip3 install swmm
import swmm
import datetime
from matplotlib.gridspec import GridSpec

# for storing models and analyzing performance
import dill as pickle # so pickle can serialize lambda functions

import os
import time


"""# load data"""

print("begin loading training data")
start_index = 9
print("start index")
print(start_index)

#with pyswmm.output.Output('/content/drive/MyDrive/SINDy/SWMM-model-reduction/AnnArborFull/full_ann_arbor_090216/full_ann_arbor.out') as out:
with pyswmm.output.Output("G:/My Drive/SINDy/SWMM-model-reduction/AnnArborFull/full_ann_arbor_090216/train.out") as out:
    # 1200 junctions, so stepping 30 loads about 40 sites

  depths = pd.DataFrame(columns= list(out.nodes.keys())[start_index:-1:30], index = out.node_series(index=0,attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH))

  #depths = pd.DataFrame(columns= list(["88-55483", "1771_PLYMOUTH_2","37035_6"]), index = out.node_series(index=0,attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH))



  for col in depths.columns:
    depths[col]= out.node_series(index=col, attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH).values()

  # all subcatchemnts have the same precip time series so just grab the first subcatchment listed
  depths['precipitation'] = out.subcatch_series(index=0, attribute=swmm.toolkit.shared_enum.SubcatchAttribute.RAINFALL).values()

# i want precipitation to be the first column
cols = depths.columns.tolist()
cols = cols[-1:] + cols[:-1]
depths = depths[cols]
df_train = depths
#depths.plot(figsize=(25,10))
#x = np.array(depths)
#x_test = np.array(depths)
print(df_train)

print("training data loaded")
print("begin loading testing data")

#with pyswmm.output.Output('/content/drive/MyDrive/SINDy/SWMM-model-reduction/AnnArborFull/full_ann_arbor_090216/test.out') as out:
with pyswmm.output.Output("G:/My Drive/SINDy/SWMM-model-reduction/AnnArborFull/full_ann_arbor_090216/test.out") as out:
  test_depths = pd.DataFrame(columns= list(out.nodes.keys())[start_index:-1:30], index = out.node_series(index=0,attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH))
  #test_depths = pd.DataFrame(columns= list(["88-55483", "1771_PLYMOUTH_2","37035_6"]), index = out.node_series(index=0,attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH))

  for col in test_depths.columns:
    test_depths[col]= out.node_series(index=col, attribute=swmm.toolkit.shared_enum.NodeAttribute.INVERT_DEPTH).values()

  # all subcatchemnts have the same precip time series so just grab the first subcatchment listed
  test_depths['precipitation'] = out.subcatch_series(index=0, attribute=swmm.toolkit.shared_enum.SubcatchAttribute.RAINFALL).values()

# i want precipitation to be the first column
cols = test_depths.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_depths = test_depths[cols]
#test_depths.plot(figsize=(25,10))
df_eval = test_depths
#x_test = np.array(test_depths)
#x = np.array(test_depths)
print(df_eval)
print("testing data loaded")

# model is initially dry
windup_timesteps = 0

# model parameters
max_polyorder = 3
max_iter = 100
max_transforms = 2

for depth_col in df_train.columns[1:]:

    # filter out junctions which have no real response in either training or testing data
    if (df_train[depth_col].max() - df_train[depth_col].min() < 0.5 
        or df_eval[depth_col].max() - df_eval[depth_col].min() < 0.5): 
        print("no response in either training or testing condition. skipping.")
        continue

    # filter out junctions with derivative greater than 50 at the first timestep
    if (abs(df_train[depth_col].diff().iloc[1]) > 50 or abs(df_eval[depth_col].diff().iloc[1]) > 50):
        print("derivative greater than 50 at first timestep. skipping.")
        continue

    # filter out junctions with depth greater than 0.01 at the first timestep (should start dry, especially after subtracting minimum)
    if (df_train[depth_col].iloc[1] > 0.01 or df_eval[depth_col].iloc[1] > 0.01):
        print("depth greater than 0.01 at first timestep. skipping.")
        continue

    # filter out junctions with depth greater than 20% of the maximum at the last timestep (should end nearly dry)
    # some junctions end with a fixed depth, which isn't physical
    if (df_train[depth_col].iloc[-1] > 0.2*df_train[depth_col].max() or df_eval[depth_col].iloc[-1] > 0.2*df_eval[depth_col].max()):
        print("depth greater than 20% of maximum at last timestep. skipping.")
        continue

    print(depth_col)
    shift=False
    for polyorder in range(1, max_polyorder+1):
        start = time.perf_counter()
        rainfall_runoff_model = modpods.delay_io_train(df_train,[depth_col],['precipitation'],windup_timesteps=windup_timesteps,
        init_transforms=1, max_transforms=max_transforms,max_iter=max_iter,
        poly_order=polyorder, verbose=False, bibo_stable=True)
        end = time.perf_counter()
        training_time_minutes = (end-start)/60


        if (shift):
            results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/swmm_modpods_results/" + str(depth_col) + '_po_'+  str(polyorder) + '_shift/')
        else:
            results_folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/swmm_modpods_results/" + str(depth_col) +'_po_'+ str(polyorder) + '_no_shift/')
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
            ax[i].plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[i+1]['final_model']['response'][depth_col][windup_timesteps+1:],label='observed')
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
            ax[i].plot(df_eval.index[windup_timesteps+1:],df_eval[depth_col][windup_timesteps+1:],label='observed')
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
