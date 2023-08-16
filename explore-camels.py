import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt

# load in all the text files in the directory: G:/My Drive/PhD Admin and Notes/paper1/CAMELS
vege = pd.read_csv("G:/My Drive/PhD Admin and Notes/paper1/CAMELS/camels_vege.txt",sep=';')

print(vege)

# make a list of zero padded strings of the huc2 codes which are 01 through 18
huc2 = [str(i).zfill(2) for i in range(1,19)]
print(huc2)
swi_and_obs_runoff = pd.DataFrame()


# grab surface water input, modeled runoff, and observed runoff for each huc2 region
for region in huc2:
  print(region)
  # iterate over the files that end in "05_model_output.txt"
  folder_path = str("G:/My Drive/PhD Admin and Notes/paper1/CAMELS/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output/flow_timeseries/daymet/" + str(region) + "/")
  print(folder_path)

  for subdir, dirs, files in os.walk(folder_path):
      for file in files:
          if ("05_model_output.txt" in str(os.path.join(subdir,file))):
              print(file)
              # read in the file
              df = pd.read_csv(str(os.path.join(subdir,file)), sep='\s+')
              print(df)
              # combine the columns YR, MNTH, DY, and YR into a single datetime column
              df.rename({'YR':'year','MNTH':'month','DY':'day','HR':'hour'},axis=1,inplace=True)
              df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])

              # set the index to the datetime column
              df.set_index('datetime',inplace=True)
              # drop all columns except for RAIM and OBS_RUN
              df = df[['RAIM','MOD_RUN','OBS_RUN']]
              print(df)
              site_id = str(file)[0:8]

              df.plot()
              plt.show()
              plt.close('all')
              



# test out modpods module and make sure it works as desired



              
