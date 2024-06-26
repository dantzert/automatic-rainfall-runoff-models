# looks very similar to the camles script 
# except the catchment attributres are different
import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
pd.set_option("display.precision", 2)


# print current working directory
print(os.getcwd())
# set current working directory to "C:/automatic-rainfall-runoff-models/for_june2024_revision/revisions-code"
os.chdir("C:/automatic-rainfall-runoff-models/for_june2024_revision/revisions-code")
print(os.getcwd())


# grab site attributes from the usgs rest api (only needs to be done once)

'''
# load sites from text file
with open("usgs_modpods_sites.txt",'r') as fp:
    sites = fp.read().splitlines()

attributes = pd.DataFrame()
for site_id in sites:
    #print("\n" + str(site_id) + "\n")
    request_string = str("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + str(site_id) + "&siteOutput=expanded&siteStatus=all")
    request_string = request_string.replace(" ","")
    print(request_string)

    response = pd.read_csv(request_string, sep='\t', comment = '#', header=[0])
    response = response[1:] # get rid of first row (data type)
    #print(response.columns)
    response.set_index('site_no',inplace=True)
    # subset of hydrologically relevant attributes
    response = response[['site_tp_cd','dec_lat_va','dec_long_va',
                         'state_cd','alt_va','huc_cd','basin_cd','topo_cd',
                         'drain_area_va','contrib_drain_area_va']]
    #print(response)
    if attributes.empty:
        # initialize attributes dataframe
        attributes = response
    else:
        # append to attributes dataframe
        attributes = pd.concat((attributes,response),axis='index')


print(attributes)

# save attributes to csv
attributes.to_csv("usgs_modpods_attributes.csv")
'''
# load attributes from csv
attributes = pd.read_csv("usgs_modpods_attributes.csv",index_col=0)
print(attributes)
# for grabbing model performance and parameters
folder_path = str('usgs_modpods_results')
shifted_eval = dict()
# below is (nearly) verbatim from camels_parameter_correlation.py


# separate results by polynomial order and number of transformations
p1t1 = pd.DataFrame(np.empty((0, 11  ) ) ) 
p1t2 = pd.DataFrame(np.empty((0, 11  ) ) )
p2t1 = pd.DataFrame(np.empty((0, 11  ) ) )
p2t2 = pd.DataFrame(np.empty((0, 11  ) ) )
p3t1 = pd.DataFrame(np.empty((0, 11  ) ) )
p3t2 = pd.DataFrame(np.empty((0, 11  ) ) )

# walk over results grabbing error metrics and model parameters
print("grabbing error metrics")
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        
        # grabbing error metrics
        if("error_metrics" in str(os.path.join(subdir, file))):
          # only look at the shifted models for now
          if ("eval" in str(os.path.join(subdir, file))):
            if ("shift" in str(os.path.join(subdir, file))):
              # the site id is 8 integers that occur immediately after usgs_moidpods_results/ and ends with an underscore
              # grab the substring coming after the first slash in the file name
              site_id = str(subdir).split("\\")[1]
              site_id = site_id.split("_")[0]

              #print(site_id)
              shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
              if ("po_1" in str(os.path.join(subdir, file))):
                if p1t1.empty:
                    p1t1.columns = shifted_eval[site_id].columns 
                p1t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                try:
                    if p1t2.empty:
                        p1t2.columns = shifted_eval[site_id].columns
                    p1t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                except:
                    pass
              if ("po_2" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
                #print(shifted_eval[site_id])
                try:
                    if p2t1.empty:
                        p2t1.columns = shifted_eval[site_id].columns
                    p2t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                except:
                    pass # that model config wasn't trained, not an error
                try:
                    if p2t2.empty:
                        p2t2.columns = shifted_eval[site_id].columns
                    p2t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                except:
                    pass
              if ("po_3" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)))
                #print(shifted_eval[site_id])
                try:
                    if p3t1.empty:
                        p3t1.columns = shifted_eval[site_id].columns
                    p3t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                except:
                    pass # that model config wasn't trained, not an error
                try:
                    if p3t2.empty:
                        p3t2.columns = shifted_eval[site_id].columns
                    p3t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                except:
                    pass
print(p1t1)
p1t1[['auto1','instant1','tr1_1','tr1_shape','tr1_scale','tr1_loc','tr1_tP','tr1_t50']] = np.nan
p1t2[['auto1','instant1','tr1_1','tr2_1','tr1_shape',
      'tr1_scale','tr1_loc','tr1_tP','tr1_t50',
      'tr2_shape','tr2_scale','tr2_loc','tr2_tP','tr2_t50']] = np.nan
# might not interpret model parameters for any higher order models than these
print("grabbing model parameters")
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        
        # grabbing model parameters
        if ("rainfall_runoff_model" in str(os.path.join(subdir, file)) and "shift" in str(os.path.join(subdir, file)) ):
            # open the file using pickle
            with open(str(os.path.join(subdir, file)), 'rb') as f:
                model = pickle.load(f)
            #print(model)
            if ("po_1" in str(os.path.join(subdir, file))):
                site_id = str(subdir).split("\\")[1]
                site_id = site_id.split("_")[0]
                #print(site_id)
                #print(str(os.path.join(subdir, file)))
                # coefficient terms for p1t1
                #print(model[1]['final_model']['model'].coefficients())
                auto1 = model[1]['final_model']['model'].coefficients()[0][0]
                instant1 = model[1]['final_model']['model'].coefficients()[0][1]
                tr1_1 = model[1]['final_model']['model'].coefficients()[0][2]
                # transformation parameters for p1t1
                tr1_shape = model[1]['shape_factors'].iloc[0,0]
                tr1_scale = model[1]['scale_factors'].iloc[0,0]
                tr1_loc = model[1]['loc_factors'].iloc[0,0]
                tr1_tP = (tr1_shape - 1)/(1/tr1_scale) + tr1_loc
                # the scale parameter in scipy.stats.gamma is the inverse of beta
                tr1_t50 = (tr1_shape)/(1/tr1_scale) + tr1_loc
                p1t1.loc[site_id,'auto1'] = auto1
                p1t1.loc[site_id,'instant1'] = instant1
                p1t1.loc[site_id,'tr1_1'] = tr1_1
                p1t1.loc[site_id,'tr1_shape'] = tr1_shape
                p1t1.loc[site_id,'tr1_scale'] = tr1_scale
                p1t1.loc[site_id,'tr1_loc'] = tr1_loc
                p1t1.loc[site_id,'tr1_tP'] = tr1_tP
                p1t1.loc[site_id,'tr1_t50'] = tr1_t50
                #print(p1t1)
                # coefficient terms for p1t2
                #print(model[2]['final_model']['model'].coefficients())
                auto1 = model[2]['final_model']['model'].coefficients()[0][0]
                instant1 = model[2]['final_model']['model'].coefficients()[0][1]
                tr1_1 = model[2]['final_model']['model'].coefficients()[0][2]
                tr2_1 = model[2]['final_model']['model'].coefficients()[0][3]
                # transformation parameters for p1t2
                tr1_shape = model[2]['shape_factors'].iloc[0,0]
                tr1_scale = model[2]['scale_factors'].iloc[0,0]
                tr1_loc = model[2]['loc_factors'].iloc[0,0]
                tr1_tP = (tr1_shape - 1)/(1/tr1_scale) + tr1_loc
                tr1_t50 = (tr1_shape)/(1/tr1_scale) + tr1_loc
                tr2_shape = model[2]['shape_factors'].iloc[1,0]
                tr2_scale = model[2]['scale_factors'].iloc[1,0]
                tr2_loc = model[2]['loc_factors'].iloc[1,0]
                tr2_tP = (tr2_shape - 1)/(1/tr2_scale) + tr2_loc
                tr2_t50 = (tr2_shape)/(1/tr2_scale) + tr2_loc
                p1t2.loc[site_id,'auto1'] = auto1
                p1t2.loc[site_id,'instant1'] = instant1
                p1t2.loc[site_id,'tr1_1'] = tr1_1
                p1t2.loc[site_id,'tr2_1'] = tr2_1
                p1t2.loc[site_id,'tr1_shape'] = tr1_shape
                p1t2.loc[site_id,'tr1_scale'] = tr1_scale
                p1t2.loc[site_id,'tr1_loc'] = tr1_loc
                p1t2.loc[site_id,'tr1_tP'] = tr1_tP
                p1t2.loc[site_id,'tr1_t50'] = tr1_t50
                p1t2.loc[site_id,'tr2_shape'] = tr2_shape
                p1t2.loc[site_id,'tr2_scale'] = tr2_scale
                p1t2.loc[site_id,'tr2_loc'] = tr2_loc
                p1t2.loc[site_id,'tr2_tP'] = tr2_tP
                p1t2.loc[site_id,'tr2_t50'] = tr2_t50
                #print(p1t2)


                


print("p1t1")
print(p1t1)
print("p1t2")
print(p1t2)
print("p2t1")
print(p2t1)

# add catchment characteristics to the dataframe
# and then drop the rows of p1t1 with nans in the first nine columns
print(p1t1)
print(attributes)
# name the index column of p1t1 to match the index column of attributes
p1t1.index.name = 'site_no'
# cast the p1t1 index to an int
p1t1.index = p1t1.index.astype(int)
p1t2.index = p1t2.index.astype(int)
p1t1 = pd.concat((p1t1,attributes),axis=1)
#p1t1 = p1t1.dropna(subset=p1t1.columns[0:17])
p1t1 = p1t1.dropna(subset='NSE')
print(p1t1)
p1t2 = pd.concat((p1t2, attributes), axis=1)
#p1t2 = p1t2.dropna(subset=+		pickle	<module 'pickle' from 'C:\\Users\\dantz\\AppData\\Local\\Programs\\Python\\Python39\\lib\\pickle.py'>	module
p1t2.columns[0:23]
p1t2 = p1t2.dropna(subset='NSE')
p2t1 = pd.concat((p2t1, attributes), axis=1)
p2t1 = p2t1.dropna(subset=p2t1.columns[0:9])
p2t2 = pd.concat((p2t2, attributes), axis=1)
p2t2 = p2t2.dropna(subset=p2t2.columns[0:9])
p3t1 = pd.concat((p3t1, attributes), axis=1)
p3t1 = p3t1.dropna(subset=p3t1.columns[0:9])
p3t2 = pd.concat((p3t2, attributes), axis=1)
p3t2 = p3t2.dropna(subset=p3t2.columns[0:9])
print(p1t1)
print(p1t2)
# for viewing the correlations want a different display format
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# these indicies need to be edited because i'm trakcing training record length and time now
# but if you're grabbing training record length and time for a model that doesn't have it, you'll get an error.
# so you'll need to exclude those old results if you want to examine those correlations. 
# maybe you can examine the correlation between eval nse and training record length in a separate script
# probably want a scatter plot for that.
# drop teh column for site_tp_cd because it's a string
p1t1 = p1t1.drop(columns=['site_tp_cd','topo_cd'])

# only calculate correlations between these two groups of columns
# 1: tr1_tP, tr1_t50, instant1, auto1, NSE
# 2: dec_lat_va, dec_long_va, alt_va, contrib_drain_area_va
import scipy.stats as stats

indices_to_keep = [2,11,12,17,18,19,20,22,26]
p1t1_for_correlation = p1t1.iloc[:,indices_to_keep]

p1t1corr = p1t1.corr()


# print all columns of dataframes
pd.set_option('display.max_columns', None)

p1t1corr = p1t1corr.iloc[indices_to_keep,indices_to_keep]
print(p1t1corr)
# print the statistical significance (p-values) of the correlations
p1t1_pvalues = p1t1_for_correlation.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) 
print(p1t1_pvalues)

#print(p1t1corr.to_string())
# show the strongest 20 p1t1corr
#print("strongest correlations in p1t1")
#print(p1t1corr.abs().unstack().sort_values(ascending=False).drop_duplicates()[0:20])
# show the strongest 20 correlations in p1t1, preserving their signs
print("strongest 20 correlations in p1t1")
print(p1t1corr.unstack().sort_values(ascending=False).drop_duplicates()[:30])
print(p1t1corr.unstack().sort_values(ascending=False).drop_duplicates()[-30:])

print("full p1t1 correlation matrix")
print(p1t1corr.to_string())


p1t2corr = p1t2.corr().iloc[23:,:23]
#print("full p1t2 correlation matrix")
#print(p1t2corr.to_string())
# show the strongest 20 p1t2corr
#print("strongest correlations in p1t2")
#print(p1t2corr.abs().unstack().sort_values(ascending=False).drop_duplicates()[0:20])

#print("strongest 20 correlations in p1t2")
#print(p1t2corr.unstack().sort_values(ascending=False).drop_duplicates()[:30])
#print(p1t2corr.unstack().sort_values(ascending=False).drop_duplicates()[-30:])





