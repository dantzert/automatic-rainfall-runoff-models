import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import contextily as cx

pd.set_option("display.precision", 2)
dataset = "camels" 
if dataset == "usgs":
    usgs_results = pd.read_csv("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/usgs_modpods_results/ensemble.csv",index_col=0)
elif dataset == "camels":
    usgs_results = pd.read_csv("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/ensemble.csv",index_col=0)

sites = usgs_results.index
usgs_results['Latitude'] = np.nan
usgs_results['Longitude'] = np.nan

for site_id in sites:
    # if site_id is less than 8 characters, add zeros to the front
    site_id = str(site_id).zfill(8)
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
    print(site_id)
    print(response.dec_lat_va.values[0])
    print(response.dec_long_va.values[0])
    # if site_id has leading zeros, remove them
    site_id = int(site_id.lstrip('0'))

    usgs_results.loc[site_id,'Latitude'] = response['dec_lat_va'].values[0]
    usgs_results.loc[site_id,'Longitude'] = response['dec_long_va'].values[0]


print(usgs_results)

# use geopandas to plot the sites
# create a geodataframe from the pandas dataframe
gdf = gpd.GeoDataFrame(
usgs_results, geometry=gpd.points_from_xy(usgs_results.Longitude, usgs_results.Latitude), crs='NAD83' )

gdf = gdf.to_crs(epsg=3857)
# plot the sites
ax = gdf.plot(figsize=(20,10))
cx.add_basemap(ax)
# turn off x and y axis labels and ticks
ax.set_axis_off()
# add a title
if dataset == "usgs":
    ax.set_title('USGS Gaging Stations', fontsize='xx-large')
elif dataset == "camels":
    ax.set_title('CAMELS Stations', fontsize='xx-large')

if dataset == "usgs":
    # add a subtitle
    ax.text(0.0, 0.05, 'Stations with NSE < -1 are displayed as NSE = -1', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # wherever NSE in gdf is less than -1, set it to -1
    gdf.loc[gdf['NSE'] < -1, 'NSE'] = -1
if dataset == "camels":
    # add a subtitle
    ax.text(0.0, 0.05, 'Stations with NSE < 0 are displayed as NSE = 0', horizontalalignment='left', transform=ax.transAxes, fontsize='x-large')
    # wherever NSE in gdf is less than 0, set it to 0
    gdf.loc[gdf['NSE'] < 0, 'NSE'] = 0

# label each point with the site id
if dataset == "usgs":
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.index):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize='small')

# color the points by their NSE value and add a colorbar
if dataset == "usgs": 
    marker_size = 100
elif dataset == "camels":
    marker_size = 50
ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['NSE'], cmap='binary', s=marker_size, zorder=2)
sm = plt.cm.ScalarMappable(cmap='binary', norm=plt.Normalize(vmin=gdf['NSE'].min(), vmax=gdf['NSE'].max()))
sm._A = []
cbar = plt.colorbar(sm)
cbar.set_label('NSE', rotation=270, labelpad=20)
# add a legend
#ax.legend(loc='lower right', title='NSE', frameon=False)


# tight layout
plt.tight_layout()


# save the figure
if dataset == "usgs":
    plt.savefig("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/usgs_modpods_results/usgs-site-map.png", dpi=600)
elif dataset == "camels":
    plt.savefig("G:/My Drive/PhD Admin and Notes/paper1/revisions-code/camels_modpods_results/camels-site-map.png", dpi=600)
plt.show()


