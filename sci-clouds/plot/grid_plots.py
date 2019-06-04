import numpy as np
import pandas as pd
import xarray as xa
import matplotlib.pyplot as plt
from glob import glob
import cartopy.crs as ccrs
from matplotlib import cm

from sci-clouds.helpers import RESULTS_REPO, DATA_REPO

def plot(varible):



def plot_areas_without_predictive_power_contourf(variable, pressurelevel, countour = False, savefig = False):
    """
    variable : Which variable do you want to plot
    pressurelevel : 300,400,500,700, 850, 1000
    countour : default False plots the courtours of a(x90-x10)
    savefig : default False
    """

    PATH = "./SesonalNoNormalisation/"
    file_name = "*"+variable  +"*"+ str(pressurelevel) + "*.csv"
    files = glob(PATH+file_name)

    if len(files) != 1:
        print("Hello didn't find any files.")

    df = pd.read_csv(files[0])
    groups = df.groupby("season")

    djf = groups.get_group("DJF")
    mam = groups.get_group("MAM")
    jja = groups.get_group("JJA")
    son = groups.get_group("SON")

    LONGITUDE = np.arange(-15, 42+0.75, 0.75) #" Doesn't include the last one "
    LATITUDE = np.arange(30, 75+0.75, 0.75)

    def calc(dataframe):
        a = dataframe["a"].values
        q10 = dataframe["q10"].values
        q90 = dataframe["q90"].values
        diff = a*(q90-q10)
        plot = (dataframe["a"].values*(dataframe["q90"].values - dataframe["q10"].values)).reshape( ( len(LONGITUDE), len(LATITUDE) ) ).T
        return plot

    def calc_lower(dataframe):
        """Helper for calculating seasonal diff<0.01"""
        a = dataframe["a"].values
        q10 = dataframe["q10"].values
        q90 = dataframe["q90"].values
        diff = a*(q90-q10)
        plot = (dataframe["a"].values*(dataframe["q90"].values - dataframe["q10"].values)).reshape( ( len(LONGITUDE), len(LATITUDE) ) ).T
        plot = abs(plot) < 0.01
        return plot

    fig, axes = plt.subplots(nrows = 2, ncols=2, figsize = (20,15))

    if countour:
        function = calc
        c = "countourf"
        path = "./Results/ImagesPredictabilityCountour/"
    else:
        function = calc_lower
        c = "smaller_than_0point01"
        path = "./Results/ImagesPredictability/"

    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    cs = ax1.contourf(LONGITUDE, LATITUDE, function(djf), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax1.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax1.set_title("DJF", fontsize = 25)

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    cs = ax2.contourf(LONGITUDE, LATITUDE, function(mam), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax2.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax2.set_title("MAM", fontsize = 25)

    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    cs = ax3.contourf(LONGITUDE, LATITUDE, function(jja), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax3.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax3.set_title("JJA", fontsize = 25)

    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    cs = ax4.contourf(LONGITUDE, LATITUDE, function(son), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax4.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax4.set_title("SON", fontsize = 25)

    if savefig:
        plt.savefig(path+"Sesonal_a(x90-x10)_"+ c +"_" + variable +"_"+ str(pressurelevel) +  ".png")


def plot_property_seasonal_contourf(variable, pressurelevel, property, savefig = False):
    """
    variable : Which variable do you want to plot

    pressurelevel : 300,400,500,700, 850, 1000

    propery : min, max, median, std, a, b, q10, q90

    savefig : default False
    """

    PATH = "./SesonalNoNormalisation/"
    file_name = "*"+variable  +"*"+ str(pressurelevel) + "*.csv"
    files = glob(PATH+file_name)

    if len(files) != 1:
        print("Hello didn't find any files.")

    df = pd.read_csv(files[0])
    groups = df.groupby("season")

    djf = groups.get_group("DJF")
    mam = groups.get_group("MAM")
    jja = groups.get_group("JJA")
    son = groups.get_group("SON")

    LONGITUDE = np.arange(-15, 42+0.75, 0.75) #" Doesn't include the last one "
    LATITUDE = np.arange(30, 75+0.75, 0.75)

    fig, axes = plt.subplots(nrows = 2, ncols=2, figsize = (20,15))

    def calc(dataframe):
        return (dataframe[property].values).reshape( ( len(LONGITUDE), len(LATITUDE) ) ).T

    function = calc

    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    cs = ax1.contourf(LONGITUDE, LATITUDE, function(djf), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax1.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax1.set_title("DJF", fontsize = 25)

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    cs = ax2.contourf(LONGITUDE, LATITUDE, function(mam), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax2.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax2.set_title("MAM", fontsize = 25)

    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    cs = ax3.contourf(LONGITUDE, LATITUDE, function(jja), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax3.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax3.set_title("JJA", fontsize = 25)

    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    cs = ax4.contourf(LONGITUDE, LATITUDE, function(son), 60, transform=ccrs.PlateCarree()) # , cmap = cm.Dark2
    ax4.coastlines(resolution='50m')
    cbar = fig.colorbar(cs)
    ax4.set_title("SON", fontsize = 25)

    if savefig:
        plt.savefig(path+"Sesonal_variation_in_"+property +"_"+ variable + "_" +str(pressurelevel)+".png")
