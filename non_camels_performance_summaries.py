import scipy.stats as stats
import os
import pandas as pd
import numpy as np
pd.set_option("display.precision", 3)
import matplotlib.pyplot as plt


dataset = "usgs" #  "usgs", "swmm" or "glwa"
folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/" + dataset + "_modpods_results")

# choose to see shifted or not shifted
data_width = 9
shifted = True # True or False
data = []
df = pd.DataFrame(data)
p1t1 = pd.DataFrame(pd.np.empty((0, data_width)))
p1t2 = pd.DataFrame(pd.np.empty((0, data_width)))
p2t1 = pd.DataFrame(pd.np.empty((0, data_width)))
p2t2 = pd.DataFrame(pd.np.empty((0, data_width)))
p3t1 = pd.DataFrame(pd.np.empty((0, data_width)))
p3t2 = pd.DataFrame(pd.np.empty((0, data_width)))
performance_summaries = dict()
shifted_eval = dict()


for subdir, dirs, files in os.walk(folder_path):
    #print(subdir)
    for file in files:
        if("error_metrics" in str(os.path.join(subdir, file))):
          name_idx = str(subdir).find("results") + 8
          site_id = str(subdir)[name_idx:str(subdir).find("_",name_idx)]
          #print(site_id)
          if ("eval" in str(os.path.join(subdir, file))):
            if ( (shifted and "no_shift" not in str(os.path.join(subdir, file))) or 
                (not shifted and "no_shift" in str(os.path.join(subdir, file))) ):
              #print(str(os.path.join(subdir, file)))
              
              if ("po_1" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(site_id)
                
                #print(shifted_eval[site_id])
                if p1t1.empty:
                    p1t1.columns = shifted_eval[site_id].columns[:data_width]
                p1t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #print(p1t1)
                #try:
                if p1t2.empty:
                    p1t2.columns = shifted_eval[site_id].columns[:data_width]
                p1t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                   # pass
              elif ("po_2" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p2t1.empty:
                    p2t1.columns = shifted_eval[site_id].columns[:data_width]
                p2t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p2t2.empty:
                    p2t2.columns = shifted_eval[site_id].columns[:data_width]
                p2t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass
              elif ("po_3" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p3t1.empty:
                    p3t1.columns = shifted_eval[site_id].columns[:data_width]
                p3t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p3t2.empty:
                    p3t2.columns = shifted_eval[site_id].columns[:data_width]
                p3t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass
print("p1t1")
print(p1t1.to_string())
print("p1t2")
print(p1t2.to_string())
print("p2t1")
print(p2t1.to_string())
print("p2t2")
print(p2t2.to_string())
print("p3t1")
print(p3t1.to_string())
print("p3t2")
print(p3t2.to_string())

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


    

print("ensemble")
print(ensemble.to_string())

# save ensemble to a csv file
ensemble.to_csv(str(folder_path + '/ensemble.csv'))


#print("p1t2")
#print(p1t2.to_string())

# format for aggregated results is just a dictionary per:
# {'model': 'FUSE (904)', 'ensemble': False, 'NSE median': 0.6222475040902987, 
# 'NSE mean': 0.5824770552137079, 'failures': 9, 'alpha_nse median': 0.7830423910576705, 
# 'beta_nse median': -0.06739990536698563, 'FHV median': -21.40688261253763, 
# 'FLV median': -66.74450580149445, 'FMS median': 15.49039319897521}
p1t1_results = {'model':'p1t1','ensemble':False,'NSE median':p1t1.NSE.median(), 'NSE mean':p1t1.NSE.mean(), 
                'failures':p1t1.NSE.lt(0).sum(),'alpha_nse median':p1t1.alpha.median(), 'beta_nse median':p1t1.beta.median(),
                'FHV median':p1t1.HFV.median(),'FLV median':p1t1.LFV.median(),'FMS median':p1t1.FDC.median()}
data.append(p1t1_results)
if not p1t2.empty:
    p1t2_results = {'model':'p1t2','ensemble':False,'NSE median':p1t2.NSE.median(), 'NSE mean':p1t2.NSE.mean(), 
                    'failures':p1t2.NSE.lt(0).sum(),'alpha_nse median':p1t2.alpha.median(), 'beta_nse median':p1t2.beta.median(),
                    'FHV median':p1t2.HFV.median(),'FLV median':p1t2.LFV.median(),'FMS median':p1t2.FDC.median()}
    data.append(p1t2_results)
if not p2t1.empty:
    p2t1_results = {'model':'p2t1','ensemble':False,'NSE median':p2t1.NSE.median(), 'NSE mean':p2t1.NSE.mean(), 
                    'failures':p2t1.NSE.lt(0).sum(),'alpha_nse median':p2t1.alpha.median(), 'beta_nse median':p2t1.beta.median(),
                    'FHV median':p2t1.HFV.median(),'FLV median':p2t1.LFV.median(),'FMS median':p2t1.FDC.median()}
    data.append(p2t1_results)
if not p2t2.empty:
    p2t2_results = {'model':'p2t2','ensemble':False,'NSE median':p2t2.NSE.median(), 'NSE mean':p2t2.NSE.mean(), 
                    'failures':p2t2.NSE.lt(0).sum(),'alpha_nse median':p2t2.alpha.median(), 'beta_nse median':p2t2.beta.median(),
                    'FHV median':p2t2.HFV.median(),'FLV median':p2t2.LFV.median(),'FMS median':p2t2.FDC.median()}
    data.append(p2t2_results)
if not p3t1.empty:
    p3t1_results = {'model':'p3t1','ensemble':False,'NSE median':p3t1.NSE.median(), 'NSE mean':p3t1.NSE.mean(), 
                    'failures':p3t1.NSE.lt(0).sum(),'alpha_nse median':p3t1.alpha.median(), 'beta_nse median':p3t1.beta.median(),
                    'FHV median':p3t1.HFV.median(),'FLV median':p3t1.LFV.median(),'FMS median':p3t1.FDC.median()}
    data.append(p3t1_results)
if not p3t2.empty:
    p3t2_results = {'model':'p3t2','ensemble':False,'NSE median':p3t2.NSE.median(), 'NSE mean':p3t2.NSE.mean(), 
                    'failures':p3t2.NSE.lt(0).sum(),'alpha_nse median':p3t2.alpha.median(), 'beta_nse median':p3t2.beta.median(),
                    'FHV median':p3t2.HFV.median(),'FLV median':p3t2.LFV.median(),'FMS median':p3t2.FDC.median()}
    data.append(p3t2_results)
ensemble_results = {'model':'modpods','ensemble':True,'NSE median':ensemble.NSE.median(skipna=True), 'NSE mean':ensemble.NSE.mean(skipna=True),
                    'failures':ensemble.NSE.lt(0).sum(),'alpha_nse median':ensemble.alpha.median(skipna=True), 'beta_nse median':ensemble.beta.median(skipna=True),
                'FHV median':ensemble.HFV.median(skipna=True),'FLV median':ensemble.LFV.median(skipna=True),'FMS median':ensemble.FDC.median(skipna=True)}
data.append(ensemble_results)

df = pd.DataFrame(data)
df = df.set_index(keys=["model", "ensemble"])
pd.set_option("display.precision", 2)
omit_std = df[df.columns.drop(list(df.filter(regex='std')))]
if shifted:
    print("performance summary, shifted")
else:
    print("performance summary, not shifted")
print(omit_std.to_string())






fig, ax = plt.subplots(figsize=(6,6))
# plot the cdfs for each model
# generate the empirical cumulative distribution functions for p1t1, p1t2, p2t1, p2t2, p3t1, p3t2

# for plotting, fill the nans with -10^6
p1t1_NSE = p1t1.NSE.fillna(-10e6)
p1t2_NSE = p1t2.NSE.fillna(-10e6)
p2t1_NSE = p2t1.NSE.fillna(-10e6)
p2t2_NSE = p2t2.NSE.fillna(-10e6)
p3t1_NSE = p3t1.NSE.fillna(-10e6)
p3t2_NSE = p3t2.NSE.fillna(-10e6)
ensemble_NSE = ensemble.NSE.fillna(-10e6)

print("number of sites")
print("ensemble: ", len(ensemble_NSE))

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


ax.plot(np.sort(p1t1_NSE),np.linspace(0, 1, len(p1t1_NSE), endpoint=False),label='p1t1',color='r',linestyle='-')
if not p1t2.empty:
    ax.plot(np.sort(p1t2_NSE),np.linspace(0, 1, len(p1t2_NSE), endpoint=False),label='p1t2',color='r',linestyle='--')
if not p2t1.empty:
    ax.plot(np.sort(p2t1_NSE),np.linspace(0, 1, len(p2t1_NSE), endpoint=False),label='p2t1',color='g',linestyle='-')
if not p2t2.empty:
    ax.plot(np.sort(p2t2_NSE),np.linspace(0, 1, len(p2t2_NSE), endpoint=False),label='p2t2',color='g',linestyle='-.')
if not p3t1.empty:
    ax.plot(np.sort(p3t1_NSE),np.linspace(0, 1, len(p3t1_NSE), endpoint=False),label='p3t1',color='c',linestyle='-')
if not p3t2.empty:
    ax.plot(np.sort(p3t2_NSE),np.linspace(0, 1, len(p3t2_NSE), endpoint=False),label='p3t2',color='c',linestyle=':')
if not ensemble.empty: # this will show if different models struggle with different dynamics, or if some are just better
    ax.plot(np.sort(ensemble_NSE),np.linspace(0, 1, len(ensemble_NSE), endpoint=False),label='final',color='k',linestyle='-')


#if shifted:
#    title = str("NSE CDF for " + dataset + " (shifted)")
#else:
#    title = str("NSE CDF for " + dataset + " (not shifted)")

ax.set_xlim(-1, 1)
if dataset == "swmm":
    ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.grid(True,alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left',fontsize='xx-large')
ax.set_xlabel('Nash-Sutcliffe Efficiency', fontsize='xx-large')
ax.set_ylabel('Cumulative Density', fontsize='xx-large')
#ax.set_title(title, fontsize=18)
# add horizontal dotted lines at 0.25, 0.5, and 0.75
#ax.axhline(y=0.25, color='k', linestyle='--')
#ax.axhline(y=0.5, color='k', linestyle='--')
#ax.axhline(y=0.75, color='k', linestyle='--')
# save as an svg
plt.savefig(str(folder_path + '/NSE_CDF.svg'),format='svg',dpi=1200)
#plt.show()



# plot the best_train_NSE vs ensemble_NSE (evaluation)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.sort(ensemble_NSE),np.linspace(0, 1, len(ensemble_NSE), endpoint=False),label='Evaluation',color='b',linestyle='-')
ax.plot(np.sort(best_train_NSE.NSE), np.linspace(0,1,len(best_train_NSE.NSE) , endpoint=False) , label='Train',color='r',linestyle='-')
ax.set_xlim(-1, 1)
if dataset == "swmm":
    ax.set_xlim(0,1)
    ax.set_title("Model Reduction [EPA-SWMM]")
if dataset == "usgs":
    ax.set_title("USGS Gaging Stations")
ax.set_ylim(0,1)
ax.grid(True,alpha=0.2)
ax.legend(loc='best',fontsize='xx-large')
ax.set_xlabel('Nash Sutcliffe Efficiency', fontsize='xx-large')
ax.set_ylabel('Cumulative Density', fontsize='xx-large')
#ax.set_title(title, fontsize=18)
# save as an svg
plt.savefig(str(folder_path + '/train_vs_eval_NSE.svg'),format='svg',dpi=600)
plt.savefig(str(folder_path + '/train_vs_eval_NSE.png'),format='png',dpi=600)
plt.show()
#plt.close('all')

# make a histogram of the column "model" within the dataframe ensemble and save it as a png
fig, ax = plt.subplots(figsize=(6,6))
ensemble['model'].value_counts().plot.bar()
# make the x axis tick labels horizontal
ax.tick_params(axis='x',labelrotation=0)
ax.set_xlabel('Model Configuration', fontsize='xx-large')
ax.set_ylabel('Count', fontsize='xx-large')
if dataset == "swmm":
    ax.set_title("Model Reduction [EPA-SWMM]",fontsize='xx-large')
if dataset == "usgs":
    ax.set_title("USGS Gaging Stations",fontsize='xx-large')
plt.savefig(str(folder_path + '/model_distribution.svg'),format='svg',dpi=600)
plt.savefig(str(folder_path + '/model_distribution.png'),format='png',dpi=600)
plt.show()
