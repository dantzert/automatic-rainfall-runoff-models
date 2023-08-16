import matplotlib.pyplot as plt   
import pandas as pd
from matplotlib.lines import Line2D
# consult: https://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1/ 

# excluding snowmelt from these parameter and input numbers because modpods is taking water surface input as the sole input

# Sacramento soil moisture accounting model, per https://www.sciencedirect.com/science/article/abs/pii/S0022169405005482?via%3Dihub
# and https://journals.ametsoc.org/view/journals/hydr/18/8/jhm-d-16-0284_1.xml
# parameters: 16
# inputs: 2, potential evapotranspiration and water surface input (rain plus snowmelt)
# 95th percentile NSE: 0.81

# Variable Infiltration Capacity (VIC) model, per https://journals.ametsoc.org/view/journals/hydr/18/8/jhm-d-16-0284_1.xml
# and https://vic.readthedocs.io/en/vic.5.0.0/Documentation/Drivers/Image/Params/ 
# parameters: about 62, very difficult to determine exact number because model configurations can be highly variable
# inputs: 7 (precip, air temp, wind speed, longwave, shortwave, atmospheric pressure, and vapor pressure)
# 95th percentile NSE: 0.77

# HBV, per https://eprints.ncl.ac.uk/file_store/production/246998/A084BCF1-F4EA-4EDF-AE6D-9E85C27A9DC4.pdf
# parameters: 15
# inputs: 3, precipitation, temperature, and evaporation
# 95th percentile NSE: 0.76 - 0.84 (use the higher one for score coloring)

# mHm, per https://hess.copernicus.org/articles/23/2601/2019/ and https://github.com/danklotz/mhm/blob/master/doc/mhm_manual_v5.8.pdf
# parameters: 83 
# inputs: 9
# 95th percentile NSE: 0.82

# EA-LSTM 8 member ensemble, per https://hess.copernicus.org/articles/23/5089/2019/
# parameters: 12936
# inputs: 5 (precip, min air temp, max air temp, shortwave, and vapor pressure)
# 95th percentile NSE: 0.89

# modpods
# parameters: 18
# inputs: 1, surface water input (rain plus snowmelt)
# 95th percentile NSE: 0.68

comparison = pd.DataFrame(index=['SAC-SMA','VIC', 'HBV', 'mHm', 'EA-LSTM', 'modpods'], columns=['Parameters', 'Inputs', 'NSE'])
comparison.loc['SAC-SMA'] = [16, 2, 0.81]
comparison.loc['VIC'] = [62, 7, 0.77]
comparison.loc['HBV'] = [15, 3, 0.84]
comparison.loc['mHm'] = [83, 9, 0.82]
comparison.loc['EA-LSTM'] = [12936, 5, 0.89]
comparison.loc['modpods'] = [18, 1, 0.68]
process_based_models = comparison.loc[['SAC-SMA','VIC', 'HBV', 'mHm']]
data_driven_models = comparison.loc[['EA-LSTM', 'modpods']]

# make an annotated scatter plot with parameters on the x axis, inputs on the y axis, and NSE as the color
fig, ax = plt.subplots(figsize=(10,5))
#ax.scatter(comparison['Inputs'],comparison['Parameters'], c=comparison['NSE'], cmap='Greys')
# add a colorbar
cbar = plt.colorbar(ax.scatter(process_based_models['Inputs'],process_based_models['Parameters'],s=200, c=process_based_models['NSE'], vmin=0.6,vmax=1.0,marker='o') , location='top') # min/max defined
ax.scatter(data_driven_models['Inputs'],data_driven_models['Parameters'],s=200, c=data_driven_models['NSE'], vmin=0.6,vmax=1.0,marker='s')

cbar.set_label('95th Percentile NSE',fontsize='xx-large')
# make the colorbar ticks only have two numbers past the decimal
cbar.set_ticks([0.6,0.70,0.8,0.9,1.0])

ax.set_ylabel('Model Parameters',fontsize='xx-large')
ax.set_xlabel('Number of Required Inputs',fontsize='xx-large')

# plot empty blue squares on the coordinates of modpods and EA-LStM
dd = ax.scatter(-10,-10, marker='s', s=200, facecolors='none', edgecolors='k',linewidth=3,label='Data-Driven')
# plot empty red circles on the coordinates of the other models
pb = ax.scatter(-10,-10, marker='o', s=200, facecolors='none', edgecolors='k',linewidth=3,label='Process-Based')
lns = [dd,pb]

for i, txt in enumerate(comparison.index):
    if txt == 'modpods':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i]-0.3 ,comparison['Parameters'][i]+25),fontsize='xx-large')
    elif txt == 'HBV':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] +0.2,comparison['Parameters'][i]-3),fontsize='xx-large')
    elif txt == 'SAC-SMA':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] +0.2,comparison['Parameters'][i]+7),fontsize='xx-large')
    elif txt == 'mHm':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] - 1,comparison['Parameters'][i]+40),fontsize='xx-large')
    elif txt == 'EA-LSTM':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] + 0.2,comparison['Parameters'][i]-1000),fontsize='xx-large')
    else:
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] + 0.2,comparison['Parameters'][i]),fontsize='xx-large')

labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left',fontsize='xx-large')
# hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yscale('log')
# set the x axis limits
ax.set_xlim(0.5,9.5)
ax.set_ylim(10,2*10**4)


# save as a png and svg "G:\My Drive\PhD Admin and Notes\paper1\revisions-code\camels_modpods_results\train_vs_eval_NSE.png"
plt.savefig('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/model_comparison_95.png',dpi=600,bbox_inches='tight')
plt.savefig('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/model_comparison_95.svg',dpi=600,bbox_inches='tight')




# different version, showing maximum NSE
comparison = pd.DataFrame(index=['SAC-SMA','VIC', 'HBV', 'mHm', 'EA-LSTM', 'modpods'], columns=['Parameters', 'Inputs', 'NSE'])
comparison.loc['SAC-SMA'] = [16, 2, 0.90]
comparison.loc['VIC'] = [62, 7, 0.88]
comparison.loc['HBV'] = [15, 3, 0.91]
comparison.loc['mHm'] = [83, 9, 0.92]
comparison.loc['EA-LSTM'] = [12936, 5, 0.96]
comparison.loc['modpods'] = [18, 1, 0.89]
process_based_models = comparison.loc[['SAC-SMA','VIC', 'HBV', 'mHm']]
data_driven_models = comparison.loc[['EA-LSTM', 'modpods']]

# make an annotated scatter plot with parameters on the x axis, inputs on the y axis, and NSE as the color
fig, ax = plt.subplots(figsize=(10,5))
#ax.scatter(comparison['Inputs'],comparison['Parameters'], c=comparison['NSE'], cmap='Greys')
# add a colorbar
cbar = plt.colorbar(ax.scatter(process_based_models['Inputs'],process_based_models['Parameters'],s=200, c=process_based_models['NSE'], vmin=0.8,vmax=1.0,marker='o') , location='top') # min/max defined
ax.scatter(data_driven_models['Inputs'],data_driven_models['Parameters'],s=200, c=data_driven_models['NSE'], vmin=0.8,vmax=1.0,marker='s')

cbar.set_label('Maximum NSE',fontsize='xx-large')
# make the colorbar ticks only have two numbers past the decimal
cbar.set_ticks([0.8,0.9,1.0])

ax.set_ylabel('Model Parameters',fontsize='xx-large')
ax.set_xlabel('Number of Required Inputs',fontsize='xx-large')

# plot empty blue squares on the coordinates of modpods and EA-LStM
dd = ax.scatter(-10,-10, marker='s', s=200, facecolors='none', edgecolors='k',linewidth=3,label='Data-Driven')
# plot empty red circles on the coordinates of the other models
pb = ax.scatter(-10,-10, marker='o', s=200, facecolors='none', edgecolors='k',linewidth=3,label='Process-Based')
lns = [dd,pb]

for i, txt in enumerate(comparison.index):
    if txt == 'modpods':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i]-0.3 ,comparison['Parameters'][i]+25),fontsize='xx-large')
    elif txt == 'HBV':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] +0.2,comparison['Parameters'][i]-3),fontsize='xx-large')
    elif txt == 'SAC-SMA':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] +0.2,comparison['Parameters'][i]+7),fontsize='xx-large')
    elif txt == 'mHm':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] - 1,comparison['Parameters'][i]+40),fontsize='xx-large')
    elif txt == 'EA-LSTM':
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] + 0.2,comparison['Parameters'][i]-1000),fontsize='xx-large')
    else:
        ax.annotate(str(txt + " (" + str(comparison['NSE'][i]) + ")"), (comparison['Inputs'][i] + 0.2,comparison['Parameters'][i]),fontsize='xx-large')

labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left',fontsize='xx-large')
# hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_yscale('log')
# set the x axis limits
ax.set_xlim(0.5,9.5)
ax.set_ylim(10,2*10**4)

# save as a png and svg "G:\My Drive\PhD Admin and Notes\paper1\revisions-code\camels_modpods_results\train_vs_eval_NSE.png"
plt.savefig('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/model_comparison_max.png',dpi=600,bbox_inches='tight')
plt.savefig('G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/model_comparison_max.svg',dpi=600,bbox_inches='tight')

plt.show()
plt.close('all')

