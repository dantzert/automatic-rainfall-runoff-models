# not doing performance summary in this file, that already has a file
# this is about correlations between model parameters and catchment attributes

# Imports
import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
pd.set_option("display.precision", 2)
folder_path = str('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results')

shifted_eval = dict()

# index: site_id

# COLUMNS:
# model parameters to relate to watershed characteristics:
# performance (by NSE)
# transformation parameters - alpha, beta, delay
# derived parameters - time to peak, T_50
# autocoreelation coeffidients 
# transformation coefficietns - don't anticipate these being as interpretable 
# catchment characteristics - area, elevation, slope, etc.
# 
attribute_names = ['clim','geol','hydro','soil','topo','vege']
attributes = pd.read_csv(str("G:/My Drive/PhD Admin and Notes/paper1/CAMELS/camels_" + attribute_names[0] + ".txt"),sep=';',index_col='gauge_id')
for idx in range(1,len(attribute_names)):
    temp = pd.read_csv(str("G:/My Drive/PhD Admin and Notes/paper1/CAMELS/camels_" + attribute_names[idx] + ".txt"),sep=';',index_col='gauge_id')
    attributes = pd.concat((attributes,temp), axis=1)

print(attributes)
print(attributes.columns)
print(type(attributes.index))

# formula for number of parameters based on polynomial order and number of transformations in a SISO model
# 3m + q(2 + m)
# where m is the number of transformations and q is the polynomial order

# separate results by polynomial order and number of transformations
p1t1 = pd.DataFrame(pd.np.empty((0, 9  ) ) ) 
p1t2 = pd.DataFrame(pd.np.empty((0, 9  ) ) )
p2t1 = pd.DataFrame(pd.np.empty((0, 9  ) ) )
p2t2 = pd.DataFrame(pd.np.empty((0, 9  ) ) )
p3t1 = pd.DataFrame(pd.np.empty((0, 9  ) ) )
p3t2 = pd.DataFrame(pd.np.empty((0, 9  ) ) )

# walk over results grabbing error metrics and model parameters
for subdir, dirs, files in os.walk(folder_path):
    for file in files:

        # grabbing error metrics
        if("error_metrics" in str(os.path.join(subdir, file))):
          # there's a better way than using literal string indices here
          #print(str(subdir)[77:77+8])
          site_id = int(str(subdir)[77:77+8])
          # only look at the shifted models for now
          if ("eval" in str(os.path.join(subdir, file))):
            if ("no_shift" not in str(os.path.join(subdir, file))):
              
              
              if ("po_1" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                if p1t1.empty:
                    p1t1.columns = shifted_eval[site_id].columns[0:9] 
                p1t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                try:
                    if p1t2.empty:
                        p1t2.columns = shifted_eval[site_id].columns[0:9]
                    p1t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                except:
                    pass
              if ("po_2" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p2t1.empty:
                    p2t1.columns = shifted_eval[site_id].columns[0:9]
                p2t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p2t2.empty:
                    p2t2.columns = shifted_eval[site_id].columns[0:9]
                p2t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass
              if ("po_3" in str(os.path.join(subdir, file))):
                shifted_eval[site_id] = pd.read_csv(str(os.path.join(subdir, file)),index_col=0)
                #print(shifted_eval[site_id])
                #try:
                if p3t1.empty:
                    p3t1.columns = shifted_eval[site_id].columns[0:9]
                p3t1.loc[site_id,:] = shifted_eval[site_id].loc[1,:]
                #except:
                    #pass # that model config wasn't trained, not an error
                #try:
                if p3t2.empty:
                    p3t2.columns = shifted_eval[site_id].columns[0:9]
                p3t2.loc[site_id,:] = shifted_eval[site_id].loc[2,:]
                #except:
                    #pass

p1t1[['auto1','instant1','tr1_1','tr1_shape','tr1_scale','tr1_loc','tr1_tP','tr1_t50']] = np.nan
p1t2[['auto1','instant1','tr1_1','tr2_1','tr1_shape',
      'tr1_scale','tr1_loc','tr1_tP','tr1_t50',
      'tr2_shape','tr2_scale','tr2_loc','tr2_tP','tr2_t50']] = np.nan
# might not interpret model parameters for any higher order models than these
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        
        # grabbing model parameters
        if ("rainfall_runoff_model" in str(os.path.join(subdir, file)) and "no_shift" not in str(os.path.join(subdir, file)) ):
            #print(str(os.path.join(subdir, file)))
            site_id = int(str(subdir)[77:77+8])
            # open the file using pickle
            with open(str(os.path.join(subdir, file)), 'rb') as f:
                model = pickle.load(f)
            #print(model)
            if ("po_1" in str(os.path.join(subdir, file))):
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

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# make a dataframe called "interpret_example" which is the rows of p1t1 where tr1_1 and instant1 are both positive
interpret_example = p1t1[(p1t1['tr1_1']>0) & (p1t1['instant1']>0) & (p1t1['tr1_shape']<500)]
print(interpret_example.sort_values(by='NSE',ascending=False).iloc[:,:].to_string())

    

'''
print("p1t1")
print(p1t1)
print("p1t2")
print(p1t2)
print("p2t1")
print(p2t1)
'''
# add catchment characteristics to the dataframe
# and then drop the rows of p1t1 with nans in the first nine columns
p1t1 = pd.concat((p1t1,attributes),axis=1)
p1t1 = p1t1.dropna(subset=p1t1.columns[0:17])
p1t2 = pd.concat((p1t2, attributes), axis=1)
p1t2 = p1t2.dropna(subset=p1t2.columns[0:9])
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

# print out tahquanemenon and greenbrier
print("tahquanemenon")
print(pd.DataFrame.transpose(p1t1.loc[[4045500]]).to_string())
print("greenbrier")
print(pd.DataFrame.transpose(p1t1.loc[[3180500]]).to_string())

# for viewing the correlations want a different display format
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# show only the strongest 10 correlations in p1t1
# get a list of strings of the columns in p1t1

# these indicies need to be edited because i'm recording the trainig time now
p1t1corr = p1t1.corr().iloc[17:,:17]
#print(p1t1corr.to_string())
# show the strongest 20 p1t1corr
#print("strongest correlations in p1t1")
#print(p1t1corr.abs().unstack().sort_values(ascending=False).drop_duplicates()[0:20])
# show the strongest 20 correlations in p1t1, preserving their signs
print("strongest 20 correlations in p1t1")
print(p1t1corr.unstack().sort_values(ascending=False).drop_duplicates()[:30])
print(p1t1corr.unstack().sort_values(ascending=False).drop_duplicates()[-30:])

print("full p1t1corr")
print(p1t1corr[['tr1_tP','tr1_t50','instant1','auto1']].to_string())

# save the table p1t1corr[['instant1','auto1','tr1_tP','tr1_t50']] as a csv
p1t1corr[['instant1','auto1','tr1_tP','tr1_t50']].to_csv(str(folder_path + '/full_correlation_table.csv'))



p1t2corr = p1t2.corr().iloc[23:,:23]
#print(p1t2corr.to_string())
# show the strongest 20 p1t2corr
#print("strongest correlations in p1t2")
#print(p1t2corr.abs().unstack().sort_values(ascending=False).drop_duplicates()[0:20])

print("strongest 20 correlations in p1t2")
print(p1t2corr.unstack().sort_values(ascending=False).drop_duplicates()[:30])
print(p1t2corr.unstack().sort_values(ascending=False).drop_duplicates()[-30:])

