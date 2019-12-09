import cdsapi
import glob
path = "/home/hanna/lagrings/"

seasons = {"DJF":["12"]}

available_years = {"79-87":     ['2005']               }

# Should # North, West, South, East. Default: global
climate_zones = {'Boreal':"60/-15/20/40",
                }
counter = 0
for season, season_list in seasons.items():
    for climate_zone, cz in climate_zones.items():
        for key, value in available_years.items():
            # Checks if the file is doenloaded before
            #print(glob.glob(path+"/dataERA5/*TCC.nc"))
            files = glob.glob(path+"*TCC.nc")
            #if path+season+"_"+climate_zone+"_"+key+'_TCC.nc' in files:
                #print(path+"/dataERA5/"+season+"_"+climate_zone+"_"+key+'_TCC.nc is downloaded')
            #    counter += 1
            #    print("Have downloaded in total : " + str(counter) + " files.")
            #else:
        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels', # reanalysis-era5-pressure-levels
            {

                #'expver' = '1', # ??
            	'area'    : cz, # retrieving subarea :Europe
            	"type" : "an", # analysis
                'product_type':'reanalysis',
                'format':'netcdf',
                'day':[
                    '01','02','03',
                    '04','05','06',
                    '07','08','09',
                    '10','11','12',
                    '13','14','15',
                    '16','17','18',
                    '19','20','21',
                    '22','23','24',
                    '25','26','27',
                    '28','29','30',
                    '31'
                ],
                'month':season_list,
                'year': value,
                'variable':'total_cloud_cover', # , '2m_temperature','surface_pressure','total_cloud_cover'
                'time':[
                    '00:00','01:00','02:00','03:00',
                    '04:00','05:00','06:00',
                    '07:00','08:00','09:00',
                    '10:00','11:00','12:00',
                    '13:00','14:00','15:00',
                    '16:00','17:00','18:00',
                    '19:00','20:00','21:00',
                    '22:00','23:00'],
            "stream": "oper", # sub-daily data
            },
            path+"/"+season+"_"+climate_zone+"_"+key+'_TCC.nc')
