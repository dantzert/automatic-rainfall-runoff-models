# based on https://github.com/kratzert/ealstm_regional_modeling/blob/master/notebooks/performance.ipynb

# Imports
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from scipy.stats import wilcoxon
import numpy as np
import os

# Add repository to Python path
BASE_CODE_DIR = "G:/My Drive/PhD Admin and Notes/paper1/ealstm_regional_modeling"
sys.path.append(BASE_CODE_DIR)
from papercode.plotutils import model_draw_style, model_specs, ecdf
from papercode.evalutils import (get_run_dirs, eval_lstm_models, 
                                 eval_benchmark_models, get_pvals, 
                                 get_mean_basin_performance, get_cohens_d)
from papercode.metrics import *


print("Loaded data from pre-computed pickle file")
with open("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/all_metrics.p", "rb") as fp:
    all_metrics = pickle.load(fp)

print(type(all_metrics))


# tabular comparison
data = []
for model_type, models in all_metrics["NSE"].items():
    if model_type == "benchmarks":
        continue
    seeds = [k for k in models.keys() if "seed" in k]
    means, medians, failures = [], [], []
    for seed in seeds:
        nses = list(models[seed].values())
        means.append(np.mean(nses))
        medians.append(np.median(nses))
        failures.append(len([v for v in nses if v <= 0]))
    data_sing = {'model_type': model_draw_style[model_type]["label"], 
                 'ensemble': False, 
                 'mean': np.mean(means), 
                 'std_mean': np.std(means),
                 'median': np.mean(medians),
                 'std_median': np.std(medians),
                 'failures': np.mean(failures),
                 'std_failures': np.std(failures)}
    data.append(data_sing)
    values = list(models["ensemble"].values())
    data_ensemble = {'model_type': model_draw_style[model_type]["label"],
                   'ensemble': True,
                   'mean': np.mean(values),
                   'median': np.median(values),
                   'failures': len([v for v in values if v < 0]) }
    data.append(data_ensemble)



df = pd.DataFrame(data)
df = df.set_index(keys=["model_type", "ensemble"])
#print(df.to_string())


'''
# cdf plot
fig, ax = plt.subplots(figsize=(16,10))

for model_type, models in all_metrics["NSE"].items():
    if 'lstm' in model_type:
        # single seed
        values = list(models['seed111'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_,
                label=f"{model_draw_style[model_type]['label']} seed111",
                color=model_draw_style[model_type]["single_color"], 
                marker=model_draw_style[model_type]['marker'], 
                markevery=20, 
                linestyle=model_draw_style[model_type]['linestyle'])
        
        # ensemble seed
        values = list(models['ensemble'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_, 
                label=f"{model_draw_style[model_type]['label']} ensemble (n=8)", 
                color=model_draw_style[model_type]['ensemble_color'], 
                linestyle=model_draw_style[model_type]['linestyle'])
    
ax.set_xlim(0, 1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('NSE', fontsize=14)
ax.set_ylabel('cumulative density', fontsize=14)
ax.set_title("Effect of (not) using static catchment attributes", fontsize=18)
plt.show()
'''
# omitted section on calculating statistical significance

# compare against benchmark models

# find all basins modeled by all benchmarks
basins = frozenset(list(all_metrics["NSE"]["ealstm_NSE"]["ensemble"].keys()))
for model, results in all_metrics["NSE"]["benchmarks"].items():
    basins = basins.intersection(list(results.keys()))

folder_path = str('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results')
modpods_basins = set()
for subdir, dirs, files in os.walk(folder_path):
    #print(subdir)
    for file in files:
        if("error_metrics" in str(os.path.join(subdir, file))):
            # there's a better way than using literal string indices here
          #print(str(subdir)[77:77+8])
          if ("po_3" in str(os.path.join(subdir, file))):
              site_id = str(subdir)[77:77+8]
              modpods_basins.add(site_id)


basins = basins.intersection(modpods_basins)
print("number of basins modeled by all benchmarks and modpods:", len(basins))

# get subset of all metrics for these share basins
sub_metrics = {metric: defaultdict(dict) for metric in all_metrics.keys()}
for metric, model_metric in all_metrics.items():
    for model_type, models in model_metric.items():
        for model, results in models.items():
            sub_metrics[metric][model_type][model] = {}
            for basin, nse in results.items():
                if basin in basins:
                    sub_metrics[metric][model_type][model][basin] = nse

'''
fig, ax = plt.subplots(figsize=(16,10))

for model_type, models in sub_metrics["NSE"].items():
    if (model_type == "ealstm_NSE") or (model_type == "lstm_no_static_NSE"):
        # single seed
        values = list(models['seed111'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_,
                label=f"{model_draw_style[model_type]['label']} seed111",
                color=model_draw_style[model_type]["single_color"], 
                marker=model_draw_style[model_type]['marker'], 
                markevery=20, 
                linestyle=model_draw_style[model_type]['linestyle'])
        
        # ensemble seed
        values = list(models['ensemble'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_, 
                label=f"{model_draw_style[model_type]['label']} ensemble (n=8)", 
                color=model_draw_style[model_type]['ensemble_color'], 
                linestyle=model_draw_style[model_type]['linestyle'])
    elif model_type == "benchmarks":
        for benchmark_model, benchmark_result in models.items():
            if "conus" in benchmark_model:
                values = list(benchmark_result.values())
                bin_, cdf_ = ecdf(values)
                ax.plot(bin_,
                        cdf_, 
                        label=model_draw_style[benchmark_model]['label'], 
                        color=model_draw_style[benchmark_model]['color'], 
                        linestyle=model_draw_style[benchmark_model]['linestyle'])
    
ax.set_xlim(0, 1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('NSE', fontsize=14)
ax.set_ylabel('cumulative density', fontsize=14)
ax.set_title("Benchmarking against CONUS-wide calibrated hydrological models", fontsize=18)
plt.show()
'''


vic_count = 0
mhm_count = 0
for basin in basins:
    lstm_nse = sub_metrics["NSE"]["ealstm_NSE"]["ensemble"][basin]
    if sub_metrics["NSE"]["benchmarks"]["VIC_conus"][basin] >= lstm_nse:
        vic_count += 1
    if sub_metrics["NSE"]["benchmarks"]["mHm_conus"][basin] >= lstm_nse:
        mhm_count += 1
        
#print(f"VIC is better (or equal) than EA-LSTM ensemble mean in {vic_count}/{len(basins)} basins")
#print(f"mHm is better (or equal) than EA-LSTM ensemble mean in {mhm_count}/{len(basins)} basins")




# tabular comparison with basin-wise benchmarks
data = []
single_model = {'model': 'EA-LSTM with NSE', 'ensemble': False}
ensemble_mean = {'model': 'EA-LSTM with NSE', 'ensemble': True}
# get EA-LSTM stats for all metrics
for metric, metric_data in sub_metrics.items():
    
    # average over single models
    seeds = [k for k in metric_data["ealstm_NSE"].keys() if "seed" in k]
    seed_vals = defaultdict(list)
    for seed in seeds:
        values = list(metric_data["ealstm_NSE"][seed].values())
        seed_vals[f"{metric} median"].append(np.median(values))
        if metric == "NSE":
            seed_vals[f"{metric} mean"].append(np.mean(values))
            seed_vals["failures"].append(len([v for v in values if v <= 0]))
        single_model[f"{metric} median"] = np.mean(seed_vals[f"{metric} median"])
        single_model[f"{metric} median std"] = np.std(seed_vals[f"{metric} median"])
        if metric == "NSE":
            single_model[f"{metric} mean"] = np.mean(seed_vals[f"{metric} mean"])
            single_model[f"{metric} mean std"] = np.std(seed_vals[f"{metric} mean"])
            single_model[f"failures"] = np.mean(seed_vals["failures"])
            single_model[f"failures std"] = np.std(seed_vals["failures"])
            
    # ensemble mean
    values = list(metric_data["ealstm_NSE"]["ensemble"].values())
    ensemble_mean[f"{metric} median"] = np.median(values)
    if metric == "NSE":
        ensemble_mean["NSE mean"] = np.mean(values)
        ensemble_mean["failures"] = len([v for v in values if v <= 0])
        
data.append(single_model)
data.append(ensemble_mean)
        
# benchmark models:
for model in model_draw_style.keys():
    if "lstm" in model:
        continue
    model_data = {"model": model_draw_style[model]["label"], "ensemble": False}
    for metric, metric_data in sub_metrics.items():
        values = list(metric_data["benchmarks"][model].values())
        model_data[f"{metric} median"] = np.median(values)
        if metric == "NSE":
            model_data["NSE mean"] = np.mean(values)
            model_data["failures"] = len([v for v in values if v <= 0])
            
    data.append(model_data)
       
#print(data[-1])

# grab modpods performance data and add to table
# just grab the shifted performance for now, can double back on that later if need be
# "All statistics were calculated from the validation period of all 447 commonly modeled basins." - so eval, no training
folder_path = str('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results')
print("modpods results")
shifted_eval = dict()

# separate results by polynomial order and number of transformations
df = pd.DataFrame(data)
p1t1 = pd.DataFrame(pd.np.empty((0, 10)))
p1t2 = pd.DataFrame(pd.np.empty((0, 10)))
p2t1 = pd.DataFrame(pd.np.empty((0, 10)))
p2t2 = pd.DataFrame(pd.np.empty((0, 10)))
p3t1 = pd.DataFrame(pd.np.empty((0, 10)))
p3t2 = pd.DataFrame(pd.np.empty((0, 10)))

for subdir, dirs, files in os.walk(folder_path):
    #print(subdir)
    for file in files:
        if("error_metrics" in str(os.path.join(subdir, file))):
            # there's a better way than using literal string indices here
          #print(str(subdir)[77:77+8])
          site_id = str(subdir)[77:77+8]
          # only look at the linear models for now

          if ("eval" in str(os.path.join(subdir, file))):
            if ("no_shift" not in str(os.path.join(subdir, file))):
              #print(str(os.path.join(subdir, file)))
              
              if ("po_1" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(site_id)
                
                #print(shifted_eval[site_id])
                if p1t1.empty:
                    p1t1.columns = shifted_eval[site_id].columns 
                p1t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #print(p1t1)
                try:
                    if p1t2.empty:
                        p1t2.columns = shifted_eval[site_id].columns
                    p1t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                except:
                    pass
              elif ("po_2" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p2t1.empty:
                    p2t1.columns = shifted_eval[site_id].columns
                p2t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p2t2.empty:
                    p2t2.columns = shifted_eval[site_id].columns
                p2t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass
              elif ("po_3" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p3t1.empty:
                    p3t1.columns = shifted_eval[site_id].columns
                p3t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p3t2.empty:
                    p3t2.columns = shifted_eval[site_id].columns
                p3t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass
print("p1t1")
print(p1t1)
print("p1t2")
print(p1t2)
print("p2t1")
print(p2t1)

shifted=True
data_width=9
train_p1t1 = pd.DataFrame(pd.np.empty((0, data_width)))
train_p1t2 = pd.DataFrame(pd.np.empty((0, data_width)))
train_p2t1 = pd.DataFrame(pd.np.empty((0, data_width)))
train_p2t2 = pd.DataFrame(pd.np.empty((0, data_width)))
train_p3t1 = pd.DataFrame(pd.np.empty((0, data_width)))
train_p3t2 = pd.DataFrame(pd.np.empty((0, data_width)))
# make a dataframe that identifies the model with the best training NSE for each site
best_train_NSE = pd.DataFrame()
# make the best_NSE index the union of the indices of the p1t1, p1t2, p2t1, p2t2, p3t1, p3t2 dataframes
best_train_NSE.index = p1t1.index.union(p1t2.index).union(p2t1.index).union(p2t2.index).union(p3t1.index).union(p3t2.index)
best_train_NSE['NSE'] = np.nan
best_train_NSE['model'] = np.nan
shifted_train = dict()

for subdir, dirs, files in os.walk(folder_path):
    #print(subdir)
    for file in files:
        if("error_metrics" in str(os.path.join(subdir, file))):
          name_idx = str(subdir).find("results") + 8
          site_id = str(subdir)[name_idx:str(subdir).find("_",name_idx)]
          #print(site_id)
          if ("train" in str(os.path.join(subdir, file))):
            if ( (shifted and "no_shift" not in str(os.path.join(subdir, file))) or 
                (not shifted and "no_shift" in str(os.path.join(subdir, file))) ):
              #print(str(os.path.join(subdir, file)))
              max_so_far = -10e6
              if ("po_1" in str(os.path.join(subdir, file))):
                shifted_train[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(site_id)
                
                #print(shifted_train[site_id])
                if train_p1t1.empty:
                    train_p1t1.columns = shifted_train[site_id].columns[:data_width]
                train_p1t1.loc[site_id,:] = shifted_train[site_id].loc[1,:]
                if (train_p1t1.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p1t1.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p1t1'
                    max_so_far = train_p1t1.NSE[site_id]
                #print(p1t1)
                #try:
                if train_p1t2.empty:
                    train_p1t2.columns = shifted_train[site_id].columns[:data_width]
                train_p1t2.loc[site_id,:] = shifted_train[site_id].loc[2,:]
                if (train_p1t2.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p1t2.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p1t2'
                    max_so_far = train_p1t2.NSE[site_id]
                #except:
                   # pass
              elif ("po_2" in str(os.path.join(subdir, file))):
                shifted_train[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_train[site_id])
                #try:
                if train_p2t1.empty:
                    train_p2t1.columns = shifted_train[site_id].columns[:data_width]
                train_p2t1.loc[site_id,:] = shifted_train[site_id].loc[1,:]
                if (train_p2t1.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p2t1.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p2t1'
                    max_so_far = train_p2t1.NSE[site_id]

                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if train_p2t2.empty:
                    train_p2t2.columns = shifted_train[site_id].columns[:data_width]
                train_p2t2.loc[site_id,:] = shifted_train[site_id].loc[2,:]
                if (train_p2t2.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p2t2.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p2t2'
                    max_so_far = train_p2t2.NSE[site_id]

                #except:
                    #pass
              elif ("po_3" in str(os.path.join(subdir, file))):
                shifted_train[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_train[site_id])
                #try:
                if train_p3t1.empty:
                    train_p3t1.columns = shifted_train[site_id].columns[:data_width]
                train_p3t1.loc[site_id,:] = shifted_train[site_id].loc[1,:]
                if (train_p3t1.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p3t1.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p3t1'
                    max_so_far = train_p3t1.NSE[site_id]

                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if train_p3t2.empty:
                    train_p3t2.columns = shifted_train[site_id].columns[:data_width]
                train_p3t2.loc[site_id,:] = shifted_train[site_id].loc[2,:]
                if (train_p3t2.NSE[site_id] > max_so_far):
                    best_train_NSE.loc[site_id,'NSE'] = train_p3t2.NSE[site_id]
                    best_train_NSE.loc[site_id,'model'] = 'p3t2'
                    max_so_far = train_p3t2.NSE[site_id]

# ensemble is taking the best training NSE for each site and putting it in a dataframe
ensemble = pd.DataFrame(columns=p1t1.columns,index=p1t1.index.union(p1t2.index).union(p2t1.index).union(p2t2.index).union(p3t1.index).union(p3t2.index))
# make the best_NSE index the union of the indices of the p1t1, p1t2, p2t1, p2t2, p3t1, p3t2 dataframes
#ensemble.index = p1t1.index.union(p1t2.index).union(p2t1.index).union(p2t2.index).union(p3t1.index).union(p3t2.index)
#ensemble.columns = p1t1.columns
ensemble['model'] = np.nan

for site in best_train_NSE.index: # take the eval nse corresponding to the model with the best training nse
    try:
        if best_train_NSE.model[site] == 'p1t1':
            ensemble.loc[site,:-1] = p1t1.loc[site,:]  
            ensemble.model[site] = 'p1t1'
        elif best_train_NSE.model[site] == 'p1t2':
            ensemble.loc[site,:-1] = p1t2.loc[site,:]
            ensemble.model[site] = 'p1t2'
        elif best_train_NSE.model[site] == 'p2t1':
            ensemble.loc[site,:-1] = p2t2.loc[site,:]
            ensemble.model[site] = 'p2t1'
        elif best_train_NSE.model[site] == 'p2t2':
            ensemble.loc[site,:-1] = p2t2.loc[site,:]
            ensemble.model[site] = 'p2t2'
        elif best_train_NSE.model[site] == 'p3t1':
            ensemble.loc[site,:-1] = p3t1.loc[site,:]
            ensemble.model[site] = 'p3t1'
        elif best_train_NSE.model[site] == 'p3t2':
            ensemble.loc[site,:-1] = p3t2.loc[site,:]
            ensemble.model[site] = 'p3t2'
    except Exception as e:
        print(e)
        pass





# format for aggregated results is just a dictionary per:
# {'model': 'FUSE (904)', 'ensemble': False, 'NSE median': 0.6222475040902987, 
# 'NSE mean': 0.5824770552137079, 'failures': 9, 'alpha_nse median': 0.7830423910576705, 
# 'beta_nse median': -0.06739990536698563, 'FHV median': -21.40688261253763, 
# 'FLV median': -66.74450580149445, 'FMS median': 15.49039319897521}
p1t1_results = {'model':'p1t1','ensemble':False,'NSE median':p1t1.NSE.median(skipna=True), 'NSE mean':p1t1.NSE.mean(skipna=True), 
                'failures':p1t1.NSE.lt(0).sum(),'alpha_nse median':p1t1.alpha.median(skipna=True), 'beta_nse median':p1t1.beta.median(skipna=True),
                'FHV median':p1t1.HFV.median(skipna=True),'FLV median':p1t1.LFV.median(skipna=True),'FMS median':p1t1.FDC.median(skipna=True)}
data.append(p1t1_results)
if not p1t2.empty:
    p1t2_results = {'model':'p1t2','ensemble':False,'NSE median':p1t2.NSE.median(skipna=True), 'NSE mean':p1t2.NSE.mean(skipna=True), 
                    'failures':p1t2.NSE.lt(0).sum(),'alpha_nse median':p1t2.alpha.median(skipna=True), 'beta_nse median':p1t2.beta.median(skipna=True),
                    'FHV median':p1t2.HFV.median(skipna=True),'FLV median':p1t2.LFV.median(skipna=True),'FMS median':p1t2.FDC.median(skipna=True)}
    data.append(p1t2_results)
if not p2t1.empty:
    p2t1_results = {'model':'p2t1','ensemble':False,'NSE median':p2t1.NSE.median(skipna=True), 'NSE mean':p2t1.NSE.mean(skipna=True), 
                    'failures':p2t1.NSE.lt(0).sum(),'alpha_nse median':p2t1.alpha.median(skipna=True), 'beta_nse median':p2t1.beta.median(skipna=True),
                    'FHV median':p2t1.HFV.median(skipna=True),'FLV median':p2t1.LFV.median(skipna=True),'FMS median':p2t1.FDC.median(skipna=True)}
    data.append(p2t1_results)
if not p2t2.empty:
    p2t2_results = {'model':'p2t2','ensemble':False,'NSE median':p2t2.NSE.median(skipna=True), 'NSE mean':p2t2.NSE.mean(skipna=True), 
                    'failures':p2t2.NSE.lt(0).sum(),'alpha_nse median':p2t2.alpha.median(skipna=True), 'beta_nse median':p2t2.beta.median(skipna=True),
                    'FHV median':p2t2.HFV.median(skipna=True),'FLV median':p2t2.LFV.median(skipna=True),'FMS median':p2t2.FDC.median(skipna=True)}
    data.append(p2t2_results)
if not p3t1.empty:
    p3t1_results = {'model':'p3t1','ensemble':False,'NSE median':p3t1.NSE.median(skipna=True), 'NSE mean':p3t1.NSE.mean(skipna=True), 
                    'failures':p3t1.NSE.lt(0).sum(),'alpha_nse median':p3t1.alpha.median(skipna=True), 'beta_nse median':p3t1.beta.median(skipna=True),
                    'FHV median':p3t1.HFV.median(skipna=True),'FLV median':p3t1.LFV.median(skipna=True),'FMS median':p3t1.FDC.median(skipna=True)}
    data.append(p3t1_results)
if not p3t2.empty:
    p3t2_results = {'model':'p3t2','ensemble':False,'NSE median':p3t2.NSE.median(skipna=True), 'NSE mean':p3t2.NSE.mean(skipna=True), 
                    'failures':p3t2.NSE.lt(0).sum(),'alpha_nse median':p3t2.alpha.median(skipna=True), 'beta_nse median':p3t2.beta.median(skipna=True),
                    'FHV median':p3t2.HFV.median(skipna=True),'FLV median':p3t2.LFV.median(skipna=True),'FMS median':p3t2.FDC.median(skipna=True)}
    data.append(p3t2_results)
ensemble_results = {'model':'modpods','ensemble':True,'NSE median':ensemble.NSE.median(skipna=True), 'NSE mean':ensemble.NSE.mean(skipna=True),
                    'failures':ensemble.NSE.lt(0).sum(),'alpha_nse median':ensemble.alpha.median(skipna=True), 'beta_nse median':ensemble.beta.median(skipna=True),
                'FHV median':ensemble.HFV.median(skipna=True),'FLV median':ensemble.LFV.median(skipna=True),'FMS median':ensemble.FDC.median(skipna=True)}
data.append(ensemble_results)
    
print("ensemble median NSE")
print(ensemble.NSE.median(skipna=True))

df = pd.DataFrame(data)
df = df.set_index(keys=["model", "ensemble"])
pd.set_option("display.precision", 2)
omit_std = df[df.columns.drop(list(df.filter(regex='std')))]
omit_std = df[df.columns.drop(list(df.filter(regex='mean')))]
print(omit_std.to_string())

# again, omitted statistical significance and effect size

# NSE cdf for basin-wise calibrated benchmarks

fig, ax = plt.subplots(figsize=(6,6))

for model_type, models in sub_metrics["NSE"].items():
    if model_type == "ealstm_NSE":
        # single seed
        '''
        values = list(models['seed111'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_,
                label=f"{model_draw_style[model_type]['label']} seed111",
                color=model_draw_style[model_type]["single_color"], 
                marker=model_draw_style[model_type]['marker'], 
                markevery=20, 
                linestyle=model_draw_style[model_type]['linestyle'])
        '''
        # ensemble seed
        values = list(models['ensemble'].values())
        bin_, cdf_ = ecdf(values)
        ax.plot(bin_,
                cdf_, 
                label=f"{model_draw_style[model_type]['label']} ensemble (n=8)", 
                color=model_draw_style[model_type]['ensemble_color'], 
                linestyle=model_draw_style[model_type]['linestyle'])
    elif model_type == "benchmarks":
        for benchmark_model, benchmark_result in models.items():
            if not "conus" in benchmark_model and not "fuse" in benchmark_model:
                values = list(benchmark_result.values())
                bin_, cdf_ = ecdf(values)
                ax.plot(bin_,
                        cdf_, 
                        label=model_draw_style[benchmark_model]['label'], 
                        color=model_draw_style[benchmark_model]['color'], 
                        linestyle=model_draw_style[benchmark_model]['linestyle'])
                # exclude the FUSE models here, going to get way too busy otherwise
  


# for plotting, fill the nans with -10^6
p1t1_NSE = p1t1.NSE.fillna(-10e6)
p1t2_NSE = p1t2.NSE.fillna(-10e6)
p2t1_NSE = p2t1.NSE.fillna(-10e6)
p2t2_NSE = p2t2.NSE.fillna(-10e6)
p3t1_NSE = p3t1.NSE.fillna(-10e6)
p3t2_NSE = p3t2.NSE.fillna(-10e6)
ensemble_NSE = ensemble.NSE.fillna(-10e6)
'''
bin_, cdf_ = ecdf(np.sort(p1t1_NSE))
ax.plot(bin_,cdf_,label='p1t1',color='k',linestyle='-')
if not p1t2.empty:
    bin_, cdf_ = ecdf(np.sort(p1t2_NSE))
    ax.plot(bin_,cdf_,label='p1t2',color='k',linestyle='-.')
if not p2t1.empty:
    bin_, cdf_ = ecdf(np.sort(p2t1_NSE))
    ax.plot(bin_,cdf_,label='p2t1',color='g',linestyle='-')
if not p2t2.empty:
    bin_, cdf_ = ecdf(np.sort(p2t2_NSE))
    ax.plot(bin_,cdf_,label='p2t2',color='g',linestyle='-.')
if not p3t1.empty:
    bin_, cdf_ = ecdf(np.sort(p3t1_NSE))
    ax.plot(bin_,cdf_,label='p3t1',color='c',linestyle='-')
if not p3t2.empty:
    bin_, cdf_ = ecdf(np.sort(p3t2_NSE))
    ax.plot(bin_,cdf_,label='p3t2',color='c',linestyle='-.')
'''
if not ensemble_NSE.empty:
    bin_, cdf_ = ecdf(np.sort(ensemble_NSE))
    ax.plot(bin_,cdf_,label='modpods',color='k',linestyle='-')

# print the site_id corresponding to the 25th , 50th, and 75th percentiles for ensemble_NSE
print("Minimum: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(0.0,interpolation='nearest')] )
print("25th percentile: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.25,interpolation='lower')] )
print("25th percentile: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.25,interpolation='higher')] )
#print(ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.25,interpolation='nearest')].index[0])
#print(ensemble[str(ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.25,interpolation='nearest')].index[0])])
print("50th percentile: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.50,interpolation='lower')] )
print("50th percentile: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.50,interpolation='higher')] )
#print(ensemble[str(ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.50,interpolation='nearest')].index[0])])
print("75th percentile: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.75,interpolation='nearest')] )
print("Maximum: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(1.0,interpolation='lower')] )
print("Maximum: ", ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(1.0,interpolation='higher')] )
#print(ensemble[str(ensemble_NSE[ensemble_NSE == ensemble_NSE.quantile(.75,interpolation='nearest')].index[0])])

print("ensemble")
print(ensemble.to_string())

# save ensemble to a csv file
ensemble.to_csv(str(folder_path + '/ensemble.csv'))


ax.set_xlim(0, 1)
ax.grid(True,alpha=0.2)
ax.legend(loc='best',fontsize='large')
ax.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
ax.set_ylabel('Cumulative Density', fontsize='xx-large')
#ax.set_title("Benchmarking against basin-wise calibrated hydrological models", fontsize=18)
#ax.axhline(y=0.25, color='k', linestyle='--',alpha=0.5)
#ax.axhline(y=0.5, color='k', linestyle='--',alpha=0.5)
#ax.axhline(y=0.75, color='k', linestyle='--',alpha=0.5)
# save as an svg
plt.savefig(str(folder_path + '/NSE_CDF.svg'),format='svg',dpi=1200)
plt.show()




# plot the best_train_NSE vs ensemble_NSE (evaluation)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.sort(ensemble_NSE),np.linspace(0, 1, len(ensemble_NSE), endpoint=False),label='Evaluation',color='b',linestyle='-')
ax.plot(np.sort(best_train_NSE.NSE), np.linspace(0,1,len(best_train_NSE.NSE) , endpoint=False) , label='Train',color='r',linestyle='-')
ax.set_xlim(0, 1)
ax.set_ylim(0,1)
ax.set_title("NCAR CAMELS")
ax.grid(True,alpha=0.2)
ax.legend(loc='best',fontsize='xx-large')
ax.set_xlabel('Nash Sutcliffe Efficiency', fontsize='xx-large')
ax.set_ylabel('Cumulative Density', fontsize='xx-large')
#ax.set_title(title, fontsize=18)
# save as an svg
plt.savefig(str(folder_path + '/train_vs_eval_NSE.svg'),format='svg',dpi=600)
plt.savefig(str(folder_path + '/train_vs_eval_NSE.png'),format='png',dpi=600)
plt.show()

# make a histogram of the column "model" within the dataframe ensemble and save it as a png
fig, ax = plt.subplots(figsize=(6,6))
ensemble['model'].value_counts().plot.bar()
# make the x axis tick labels horizontal
ax.tick_params(axis='x',labelrotation=0)
ax.set_xlabel('Model Configuration', fontsize='xx-large')
ax.set_ylabel('Count', fontsize='xx-large')
ax.set_title("NCAR CAMELS",fontsize='xx-large')
plt.savefig(str(folder_path + '/model_distribution.svg'),format='svg',dpi=600)
plt.savefig(str(folder_path + '/model_distribution.png'),format='png',dpi=600)
plt.show()
