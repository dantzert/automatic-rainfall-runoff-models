# import modpods and other libraries
import sys
sys.path.append("G:/My Drive/modpods")
import modpods
#print(modpods)
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import pickle
import math
import datetime
import pandas as pd
import matplotlib.pyplot as plt   
from matplotlib.colors import LogNorm
import numpy as np
import copy
import requests
import json
import dill as pickle
import os
import time
from matplotlib.gridspec import GridSpec
import re


windup_time = datetime.timedelta(days=90) # three month windup
# should be sufficient for small catchments 
eval_time = datetime.timedelta(days=365) # one year is used for evaluation (actually nine months due to windup)

max_years = 7 # max number of years of data to use for training and eval
lookback_years = 18 # how many years we'll look backwards to find the data

area_cutoff = 100 # km2. won't train sites with larger contributing areas than this

# model parameters
max_polyorder = 3
max_iter = 100
max_transforms = 2
'''
# grab sites in warm regions (want to minimize impact of snowmelt, but if snowmelt is significant you should see a negative correlation btw score and altitude)
state_cds = ['or','ca','nv','ut','az','co','nm','tx',
             'ok','ks','mo','ar','la','ky','tn','ms',
             'al','ga','va','nc','sc','fl']
sites = list()
#state_cds = ['al']
for state in state_cds:
  # stations that measure precip and are on a body of water
  request_url = str(str("https://waterservices.usgs.gov/nwis/iv/?format=rdb&indent=on&stateCd=") + state + str("&parameterCd=00045&siteType=LK,ST,ST-CA,ST-DCH,ST-TS&siteStatus=active"))
  how_many = pd.read_csv(request_url, header=[0], skiprows=12, nrows=1)
  #print(how_many.iloc[0])
  number_of_sites = int(re.findall(r'\d+', str(how_many.iloc[0]))[0])
  #print(number_of_sites)
  meta = pd.read_csv(request_url,header=None, skiprows=14, nrows=number_of_sites, sep='\t')
  

  for i in range(number_of_sites):
    site_id = meta.iloc[i][0][10:18]
    request_string = str("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + str(site_id) + "&siteOutput=expanded&siteStatus=all")
    print(request_string)
    try:
        response = pd.read_csv(request_string, sep='\t', comment = '#', header=[0])
        response = response[1:] # get rid of first row (data type)
        if float(response['drain_area_va'].values[0]) < area_cutoff/2.59: # area smaller than XX km2
          print("added site: " + str(site_id))
          sites.append(site_id)
        else:
            print("site area greater than 500 km2. excluded: " + str(site_id))
    except:
        print("error on fetch. skipping")


print("writing ", len(sites), " site ids to file")
# write these to a text file
# the size of the contributing area and whether it collects precip won't change so only need to run the above once
with open ("usgs_modpods_sites.txt",'w') as fp:
    for site in sites:
            fp.write(str(site) + "\n")
'''

# load sites from text file
with open("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/usgs_modpods_sites.txt",'r') as fp:
    sites = fp.read().splitlines()

#print(sites)
print("number of sites evaluated for inclusion: " + str(len(sites)))


for site_id in sites[16:]:
    eval_end = datetime.datetime(month=1,day=1,year=2023,hour=0,minute=0,second=0) # filtered to active sites with rainfall measurements, so end should be recent
    train_start = (eval_end - datetime.timedelta(days=365*lookback_years)).date()
    print("\n" + str(site_id) + "\n")
    # triage sites for clean data - throw out anything with error codes in the stage column
    request_string = str("https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="+site_id+"&startDT="+str(train_start - windup_time)+"&endDT="+str(eval_end.date())+"&parameterCd=00045,00065&siteStatus=all")
    request_string = request_string.replace(" ","")
    print(request_string)
    attempts = 0
    while attempts < 10:
        print(attempts)
        try:
            meta = pd.read_csv(request_string, skiprows=14,nrows=10,sep='\t')
            break # if successful, break out of the while loop
        except:
            print("error on fetch. retrying")
            attempts += 1
            time.sleep(1)
    if attempts >= 10:
        print("no data for these criteria, skip")
        continue
    site_name = meta.columns[0][5:]
    meta_string = meta[meta.columns[0]].str.cat(sep='\n')
    print(meta_string)
    #print(meta)
    if ("Precipitation" not in meta_string or "Gage" not in meta_string):
        print("doesn't have both gage height and precip at the station, skip")
        continue
    #data = pd.read_csv(request_string,header = [0],sep='\t',nrows=24*4*365*max_years,comment = '#').dropna(axis='index')
    attempts = 0
    while attempts < 10: # we should not error here if the meta request worked
        print(attempts)
        try:
            data = pd.read_csv(request_string,header = [0,1],sep='\t',comment = '#', dtype={0:str, 1:str, 2:str, 3:str, 4:float,5:str, 6:float,7:str},parse_dates=[2]).dropna(axis='index') # get rid of rows where precip or gage height is missing
            break # if successful, break out of the while loop
        except Exception as e: 
            print(e)
            print("error on fetch. retrying")
            attempts += 1
            time.sleep(1)
    if attempts >= 10:
        print("no data for these criteria, skip")
        continue
    data.columns = data.columns.droplevel(1) # get rid of the second header row (datatype code)
    data = data[:24*4*365*max_years] #  limit total quantity of training data
   
    print(data)
    #print(data.columns)
    #print(data.iloc[:,-2])
    '''
    try:
        meta = pd.read_csv(request_string, skiprows=14,nrows=10,sep='\t')
        site_name = meta.columns[0][5:]
        meta_string = meta[meta.columns[0]].str.cat(sep='\n')
        print(meta_string)
        #print(meta)
        if ("Precipitation" not in meta_string or "Gage" not in meta_string):
            print("doesn't have both gage height and precip at the station, skip")
            continue
        data = pd.read_csv(request_string,header = [0],sep='\t',nrows=24*4*365*max_years,comment = '#').fillna(method='bfill')
        # nrows limits data to max_years, longer than that at quarterly timestep is expensive to handle
        data = data[1:] # get rid of second header row
        print(data)
        #print(data.columns)
        #print(data.iloc[:,-2])
    except:
        print("error in parsing response. skipping")
        continue
    '''
    try:
        pd.to_numeric(data.iloc[:,-2]) # if no errors, this data is clean over the entire interval
        print("looks good. no error codes in stage column. proceeding with training.")
    except:
        print("error codes in stage column")
        print("throwing out site:")
        print(site_id)
        continue # move to next site, skip this one

    # make index timezone aware
    data.set_index('datetime', inplace=True)
    data.index = pd.DatetimeIndex(data.index)#,ambiguous=dst_times)

    if (data.tz_cd.iloc[0] == "EST" or data.tz_cd.iloc[0] == "EDT"):
      usgs_tz = "US/Eastern"
    elif (data.tz_cd.iloc[0] == "CST" or data.tz_cd.iloc[0] == "CDT"):
      usgs_tz = "US/Central"
    elif (data.tz_cd.iloc[0] == "MST" or data.tz_cd.iloc[0] == "MDT"):
        usgs_tz = "US/Mountain"
    elif (data.tz_cd.iloc[0] == "PST" or data.tz_cd.iloc[0] == "PDT"):
        usgs_tz = "US/Pacific"
    else:
        print("error: unrecognized timezone")

    data = data.tz_localize(usgs_tz, ambiguous = 'NaT',nonexistent='NaT')
    data = data.tz_convert("UTC")
    data = data[~data.index.duplicated(keep='first')]
    # drop rows with equipment failures
    print(data)
    data.drop(data[data[data.columns[-2]]  == "Eqp" ].index, inplace=True)
    meta_string = meta[meta.columns[0]].str.cat(sep='\n')
    data['station_precip'] = data[data.columns[-4]].astype(float)
    data['gage_ht'] = data[data.columns[-3]].astype(float) # appended a column, so reach one further back
    #print(data.station_precip)

    quarter = pd.DataFrame()
    quarter.index = pd.date_range(data.index[0],data.index[-1],freq='15T')
    quarter['prcp_in'] = data['station_precip']#.resample('15T').sum()
    quarter['stage_ft'] = data['gage_ht']
    
    dt = pd.infer_freq(quarter.index)  
    #print("timestep = " + str(dt))
    # interpolate any missing stage measurements
    quarter.stage_ft.interpolate(inplace=True, method='time', limit_area='inside') 
    # fill na's on precip with zeros
    quarter.prcp_in.fillna(0)
    # drop any rows still with nan values
    quarter.dropna(axis=0, inplace=True) 

    
    orig_length = len(quarter)
    '''
    plt.figure(figsize=(10,5))
    quarter.stage_ft.plot()
    plt.ylabel("stage (ft)")
    quarter.prcp_in.plot(secondary_y=True)
    plt.legend()
    plt.show()
    '''
    # sometimes precip starts later than stage, so trim the beginning of the record to match
    quarter = quarter[quarter.index >= quarter[quarter.prcp_in > 0].index[0]]
    print("trimmed " + str(orig_length - len(quarter)) + " rows from beginning of record (no precip)")
    print(quarter)
    # remove constant offset in stage height
    print("constant offset in stage height: " + str(quarter.stage_ft.min()))
    quarter.stage_ft = quarter.stage_ft - quarter.stage_ft.min()                        
    print("final record")
    print(quarter)
    #continue # go back to start, just looking at data fetching for now

    '''
    plt.figure(figsize=(10,5))
    quarter.stage_ft.plot()
    plt.ylabel("stage (ft)")
    quarter.prcp_in.plot(secondary_y=True)
    plt.legend()
    plt.show()
    '''

    # NO SHIFT TRAINING then SHIFT TRAINING
    # still need to implement front vs back training. might not be necessary even though it would be more complete that way
    shift = True # False
    # only training shifted version
    for i in range(0,2):
        if (i < 1): # first time, haven't loaded data yet
            print("no shift")
            df = quarter
            if shift:
                continue
  
        else:
            print("shift is on")
            # shift the forcing back one timestep (one day) to make the system causal
            #print(df[['stage_ft','prcp_in']])
            df.prcp_in = df.prcp_in.shift(-1)
            df.dropna(inplace=True)
            #print(df[['stage_ft','prcp_in']])
            shift=True

        
        after_windup = (df.index[-1] - df.index[0]) - windup_time
        if (after_windup <= windup_time):
            print("record too short, skipping")
            continue
        
        train_start = df.index[df.index >= df.index[0] + windup_time] # once windup is complete
        train_end = df.index[df.index >= df.index[-1] - eval_time] # before eval has started
        if (train_end[0] < train_start[0]):
            print("record too short, skipping")
            print("train end: " + str(train_end))
            print("train start: " + str(train_start))
            continue
        df_train = df.loc[train_start[0]:train_end[0],:] # this means that the training period is variable
        eval_start = df.index[df.index >= df.index[-1] - eval_time] 
        eval_end = df.index[-1] # set eval period
        df_eval = df.loc[eval_start[0]:eval_end,:]
        print("train start: " + str(train_start))
        print("train end: " + str(train_end))
        print("eval start: " + str(eval_start))
        print("eval end: " + str(eval_end))
        print("training data")
        print(df_train)
        print("evaluation data")
        print(df_eval)
        training_length_days = (train_end[0] - train_start[0]).days
        # infer the frequency of df
        try:
            dt = pd.infer_freq(df.index)
        except:
            dt = df.index[1] - df.index[0] # use first timedelta if infer_freq fails
        if (dt == None):
            dt = df.index[1] - df.index[0] # use first timedelta if infer_freq fails
        windup_timesteps = int(windup_time / dt)
        print("dt")
        print(dt)
        print("windup time")
        print(windup_time)
        print("windup timesteps")
        print(windup_timesteps)
        for polyorder in range(1, max_polyorder+1):
            start = time.perf_counter()
            rainfall_runoff_model = modpods.delay_io_train(df_train, ['stage_ft'],['prcp_in'],windup_timesteps=windup_timesteps,
                    init_transforms=1, max_transforms=max_transforms,max_iter=max_iter,
                    poly_order=polyorder, verbose=False, bibo_stable=True)
            end = time.perf_counter()
            training_time_minutes = (end-start)/60

            if (shift):
                results_folder_path = str('usgs_modpods_results/' + str(site_id) + '_po_'+  str(polyorder) + '_shift/')
            else:
                results_folder_path = str('usgs_modpods_results/' + str(site_id) +'_po_'+ str(polyorder) + '_no_shift/')
            if not os.path.exists(results_folder_path):
                # Create a new directory because it does not exist
                os.makedirs(results_folder_path)
            with open(str(results_folder_path  + 'rainfall_runoff_model'),'wb') as f:
                pickle.dump(rainfall_runoff_model,f) # binary of full training results
            perf = pd.DataFrame.from_dict(rainfall_runoff_model[1]['final_model']['error_metrics'],orient='columns',dtype='float')
            for num_transforms in range(2, max_transforms+1):
                perf = pd.concat([perf, pd.DataFrame.from_dict(rainfall_runoff_model[num_transforms]['final_model']['error_metrics'])])
            perf.index = range(1, max_transforms+1)
            perf['training_length_days'] = training_length_days
            perf['training_time_minutes'] = training_time_minutes
            perf.to_csv(str(results_folder_path +'training_error_metrics.csv'))
            print(perf)
            del perf
            # plot the results
            fig, ax = plt.subplots(max_transforms,1,figsize=(10,10))
            for i in range(0,max_transforms):
                ax[i].plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[i+1]['final_model']['response']['stage_ft'][windup_timesteps+1:],label='observed')
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
            diverged_sims=list()
            eval_simulations = list() # for plotting
            for num_transforms in range(1, max_transforms+1):
                eval_sim = modpods.delay_io_predict(rainfall_runoff_model, df_eval, num_transforms,evaluation=True)
                eval_simulations.append(eval_sim['prediction'])
                diverged_sims.append(eval_sim['diverged'])
                if (num_transforms < 2):
                    eval_perf = pd.DataFrame.from_dict(eval_sim['error_metrics'],orient='columns',dtype='float')
                else:
                    eval_perf = pd.concat([eval_perf, pd.DataFrame.from_dict(eval_sim['error_metrics'])])
            print(eval_perf)
            eval_perf.index = range(1, max_transforms+1)
            eval_perf['training_length_days'] = training_length_days
            eval_perf['training_time_minutes'] = training_time_minutes
            eval_perf.to_csv(str(results_folder_path +'eval_error_metrics.csv'))
            print(eval_perf)
            del eval_perf
            # plot the results
            fig, ax = plt.subplots(max_transforms,1,figsize=(10,10))
            for i in range(0,max_transforms):
                ax[i].plot(df_eval.index[windup_timesteps+1:],df_eval['stage_ft'][windup_timesteps+1:],label='observed')
                if (not diverged_sims[i]): # simulation didn't diverge, so the simulated data is valid
                    ax[i].plot(df_eval.index[windup_timesteps+1:],eval_simulations[i],label='simulated')
                ax[i].set_title(str(str(i+1) + ' transformation(s)'))
                if (i<1):
                    ax[i].legend()
            fig.suptitle("evaluation")
            plt.tight_layout()

            plt.savefig(str(results_folder_path + "eval_viz.png"), dpi=300,bbox_inches='tight')
            plt.savefig(str(results_folder_path + "eval_viz.svg"), dpi=300,bbox_inches='tight')
            plt.close()
