import numpy as np
import cartopy as cp
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sci-clouds.helpers import FIGURE_REPO, LAT, LON

def plot_map(lat = (30,67), lon = (-15,42), path = FIGURE_REPO +"maps/",
    title = "MeteoSat vision Europa", filename = "MeteoSat_vision_Europa.png"):
    """
    Plot map defined by lat lon, default Europe.
    Path should be to lagringshotell, don't save png's on git ...
    """
    plt.figure(figsize = (15,15))
    ax = plt.axes(projection = ccrs.PlateCarree())

    ax.add_feature(cp.feature.OCEAN, zorder=0)
    ax.add_feature(cp.feature.LAND, zorder=0, edgecolor='black')
    ax.coastlines(resolution='50m')
    ax.set_extent([lon[0], lon[1], lat[0], lat[1]], ccrs.PlateCarree())

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    ax.text(0.03, 0.07, "MeteoSat vision Europa", transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    plt.savefig(path + filename, bbox_inces = "thight")
    #plt.show()
    # legend = ["Boreal", "Temperate", "Mediterranean"]
    # plt.legend(legend)
    # plt.title("Division Climate Zones", fontsize = 20)

if __name__ == "__main__":
    plot_map()
