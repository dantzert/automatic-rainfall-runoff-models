
import scipy.stats as stats
import os
import pandas as pd
import numpy as np
pd.set_option("display.precision", 3)
import matplotlib.pyplot as plt

#folder_path = str('/content/drive/MyDrive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results')
folder_path = "G:/My Drive/PhD Admin and Notes/paper1/revisions-code/glwa_modpods_results"

shifted_train = dict()
shifted_eval = dict()
noshift_train = dict()
noshift_eval = dict()
performance_summaries = dict()
for subdir, dirs, files in os.walk(folder_path):
    #print(subdir)
    for file in files:
        if("error_metrics" in str(os.path.join(subdir, file))):
          print(str(subdir)[75:75+15])
          site_id = str(subdir)[75:75+15]
          #print(str(os.path.join(subdir, file)))
          # only look at the linear models
          if ("po_1" in str(os.path.join(subdir, file))):
            if ("training" in str(os.path.join(subdir, file))):
              if ("no_shift" in str(os.path.join(subdir, file))):
                noshift_train[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
              else:
                shifted_train[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
            elif ("eval" in  str(os.path.join(subdir, file))):
              if ("no_shift" in str(os.path.join(subdir, file))):
                noshift_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
              else:
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)))

          #print(str(file))
          #print(str(os.path.join(subdir, file)))
          #site_id = str(file).partition('_')[2][:-33]
          #print(site_id)
          #performance_summaries[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
          #trained_site_ids.append(str(subdir)[-8:])
        #print(os.path.join(subdir, file))
#print(shifted_train['03439000'].NSE.mean())
#for site_id in shifted_eval:
  #print(shifted_eval[site_id])

# grab all the NSE's from each type and make a list
shift_train_NSE = list()
for site_id in shifted_train:
  shift_train_NSE.append(shifted_train[site_id].NSE.max())
shift_eval_NSE = list()
for site_id in shifted_eval:
  shift_eval_NSE.append(shifted_eval[site_id].NSE.max())
noshift_train_NSE = list()

for site_id in noshift_train:
  noshift_train_NSE.append(noshift_train[site_id].NSE.max())
noshift_eval_NSE = list()
for site_id in noshift_eval:
  noshift_eval_NSE.append(noshift_eval[site_id].NSE.max())
print(shift_train_NSE)
print(shift_eval_NSE)
print(noshift_train_NSE)
print(noshift_eval_NSE)


print("Train Failure Rate (NSE < 0)")
print("Shift: ", len([x for x in shift_train_NSE if x < 0]) / len(shift_train_NSE))
print("No Shift:", len([x for x in noshift_train_NSE if x < 0]) / len(noshift_train_NSE))

print("Eval Failure Rate (NSE < 0)")
print("Shift: ", len([x for x in shift_eval_NSE if x < 0]) / len(shift_eval_NSE))
print("No Shift:", len([x for x in noshift_eval_NSE if x < 0]) / len(noshift_eval_NSE))


print("Train Failure Rate (NSE < -100)")
print("Shift: ", len([x for x in shift_train_NSE if x < -100]) / len(shift_train_NSE))
print("No Shift:", len([x for x in noshift_train_NSE if x < -100]) / len(noshift_train_NSE))

print("Eval Failure Rate (NSE < -100)")
print("Shift: ", len([x for x in shift_eval_NSE if x < -100]) / len(shift_eval_NSE))
print("No Shift:", len([x for x in noshift_eval_NSE if x < -100]) / len(noshift_eval_NSE))

shift_train_bins = np.linspace(0,1,len(shift_train_NSE))
shift_eval_bins = np.linspace(0,1,len(shift_eval_NSE))
noshift_train_bins = np.linspace(0,1,len(noshift_train_NSE))
noshift_eval_bins = np.linspace(0,1,len(noshift_eval_NSE))
print("length of noshift eval")
print(len(noshift_eval_NSE))
print("length of shift eval")
print(len(shift_eval_NSE))
plt.figure(figsize=(10,5))
plt.plot(shift_train_bins,np.sort(shift_train_NSE),'b--',label='shift,train')
plt.plot(shift_eval_bins,np.sort(shift_eval_NSE),'r--',label='shift,eval')
plt.plot(noshift_train_bins,np.sort(noshift_train_NSE),'g-',label='noshift,train')
plt.plot(noshift_eval_bins,np.sort(noshift_eval_NSE),'y-',label='noshift,eval')
plt.ylabel("NSE")
plt.ylim([-1,1])
plt.hlines(0,0,1,'k')
plt.legend(fontsize=20)
plt.title("NSE CDF [GLWA]")
plt.show()


