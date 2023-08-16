from this import d
import scipy.stats as stats
import os
import pandas as pd
import numpy as np

dataset = "usgs" #  "usgs", "swmm", "camels", or "glwa"
folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/" + dataset + "_modpods_results")


# choose to see shifted or not shifted
data_width = 10
if dataset =="usgs":
    data_width=11 # also have training_length_days

shifted = True # True or False (needs to be correct for the dataset, usgs and camels yes, swmm no)
data = []
df = pd.DataFrame(data)
p1t1 = pd.DataFrame(pd.np.empty((0, data_width)))
p2t1 = pd.DataFrame(pd.np.empty((0, data_width)))
p3t1 = pd.DataFrame(pd.np.empty((0, data_width)))
# only need the one transformation record because training time is recroded for 1 and 2 transfomrations total
performance_summaries = dict()
shifted_eval = dict()

# try- except blocks because older training runs won't have training time reocrded

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
                if (len(pd.read_csv(str(os.path.join(subdir, file)),index_col=0).columns) > 9): # needs to have the training length
                    shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                    if p1t1.empty:
                        p1t1.columns = shifted_eval[site_id].columns[:data_width]

                    p1t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]


              elif ("po_2" in str(os.path.join(subdir, file))):
                if (len(pd.read_csv(str(os.path.join(subdir, file)),index_col=0).columns) > 9): # needs to have the training length
                    shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)

                    if p2t1.empty:
                        p2t1.columns = shifted_eval[site_id].columns[:data_width]

                    p2t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]

              elif ("po_3" in str(os.path.join(subdir, file))):
                if (len(pd.read_csv(str(os.path.join(subdir, file)),index_col=0).columns) > 9): # needs to have the training length
                    shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)

                    if p3t1.empty:
                        p3t1.columns = shifted_eval[site_id].columns[:data_width]

                    p3t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
 

print("p1t1")
print(p1t1.to_string())
print("p2t1")
print(p2t1.to_string())
print("p3t1")
print(p3t1.to_string())

# make a new datafram whose index is the intersectino of the indices of p1t1, p2t1, and p3t1
training_times = pd.DataFrame(index=p1t1.index.intersection(p2t1.index).intersection(p3t1.index), columns=["p1t1","p2t1","p3t1"])
training_times["p1t1"] = p1t1["training_time_minutes"]
training_times["p2t1"] = p2t1["training_time_minutes"]
training_times["p3t1"] = p3t1["training_time_minutes"]
training_times['total'] = training_times.sum(axis=1)

print(training_times.to_string())

# get the mean and standard deviation of the training times
print("mean")
print(training_times.mean(axis=0))
print("std")
print(training_times.std(axis=0))

# calculate and print the 25th, 50th, and 75th percentiles
print("25th")
print(training_times.quantile(q=0.25))
print("50th")
print(training_times.quantile(q=0.50))
print("75th")
print(training_times.quantile(q=0.75))









