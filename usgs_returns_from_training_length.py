import scipy.stats as stats
import os
import pandas as pd
import numpy as np
pd.set_option("display.precision", 3)
import matplotlib.pyplot as plt


dataset = "usgs" 
folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/" + dataset + "_modpods_results")

# choose to see shifted or not shifted
data_width = 11 # include training_length_days and training_time_minutes
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

# drop rows which have NaNs
ensemble.dropna(axis=0,how='any',inplace=True)
p1t1.dropna(axis=0,how='any',inplace=True)
p1t2.dropna(axis=0,how='any',inplace=True)
p2t1.dropna(axis=0,how='any',inplace=True)
p2t2.dropna(axis=0,how='any',inplace=True)
p3t1.dropna(axis=0,how='any',inplace=True)
p3t2.dropna(axis=0,how='any',inplace=True)

    

print("ensemble")
print(ensemble.to_string())

print("p1t1 correlations")
print(p1t1.corr().iloc[9,:])
print("p1t2 correlations")
print(p1t2.corr().iloc[9,:])
print("p2t1 correlations")
print(p2t1.corr().iloc[9,:])
print("p2t2 correlations")
print(p2t2.corr().iloc[9,:])
print("p3t1 correlations")
print(p3t1.corr().iloc[9,:])
print("p3t2 correlations")
print(p3t2.corr().iloc[9,:])
# print the correlation between training_length_days and every other column for ensemble
print("ensemble correlations")
print(ensemble.iloc[:,:-1].corr(numeric_only=False).iloc[9,:])

# drop rows in ensemble which have NSE less than -1 (decent performance)
ensemble = ensemble[ensemble.NSE > -2]

# correlation between training_length_days and NSE for ensemble
r = ensemble.iloc[:,:-1].corr(numeric_only=False).iloc[9,2] 


# make a scatter plot of training_length_days vs NSE for each model
fig, ax = plt.subplots()
#ax.scatter(p1t1.training_length_days,p1t1.NSE,label='p1t1')
#ax.scatter(p1t2.training_length_days,p1t2.NSE,label='p1t2')
#ax.scatter(p2t1.training_length_days,p2t1.NSE,label='p2t1')
#ax.scatter(p2t2.training_length_days,p2t2.NSE,label='p2t2')
#ax.scatter(p3t1.training_length_days,p3t1.NSE,label='p3t1')
#ax.scatter(p3t2.training_length_days,p3t2.NSE,label='p3t2')
ax.scatter(ensemble.training_length_days/365,ensemble.NSE,label='ensemble')
ax.set_xlabel('Training Record Length [Years]',fontsize='large')
ax.set_ylabel('Evaluation Nash Sutcliffe Efficiency',fontsize='large')
ax.set_title("Models with NSE < -2 excluded",fontsize='large')
#ax.set_xlim(0, 3000)
#ax.set_ylim(-10,1)
# plot the line of best fit between ensemble.training_length_days and ensemble.NSE
training_days = ensemble.training_length_days.values.astype(float)
nse = ensemble.NSE.values.astype(float)
m, b = np.polyfit(training_days, nse, 1)
print("gain in NSE per extra year of training")
print(m*365)
line_label = str('Best Fit | r = ' + str(np.round(r,decimals=2)) + ' | NSE/year = ' + str(np.round(m*365,decimals=2)))
ax.plot(training_days/365, m*training_days + b, color='red',label=line_label)
ax.legend(loc='lower center',fontsize='large')
plt.tight_layout()
plt.savefig(str(folder_path + '/returns_from_training_length.png'),format='png',dpi=600)
plt.show()


