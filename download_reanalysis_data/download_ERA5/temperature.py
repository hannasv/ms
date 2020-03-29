import cdsapi
import glob

path = "//uio/lagringshotell/geofag/students/metos/hannasv/dataERA5_grib/"

seasons = {"DJF":["12","01","02"],
            "MAM":['03','04','05'],
            "JJA":['06','07' ,'08'],
            "SON":['09','10','11']}

available_years = {"79-87":     ['1979','1980','1981',
                                '1982','1983','1984',
                                '1985','1986','1987'],
                    "88-94":    ['1988','1989','1990',
                                '1991','1992','1993',
                         	    '1994'],
                    "95-03": ['1995','1996','1997','1998',
                            '1999','2000','2001','2002',
             	              '2003'],
                    "04-10":['2004','2005',
             	                '2006','2007','2008', '2009','2010'],
                    "11-18":['2011',
            	           '2012','2013','2014',
                           '2015','2016','2017',
            	            '2018']
                    }

# Should # North, West, South, East. Default: global
climate_zones = {'Boreal':"75/-15/57/42",
                'Temperate':"56.75/-15/46/42",
                'Mediterranean': "45.75/-15/30/42"
                }
counter = 0
for season, season_list in seasons.items():
    for climate_zone, cz in climate_zones.items():
        for key, value in available_years.items():
            files = glob.glob(path+"*T.nc")

            if path+season+"_"+climate_zone+"_"+key+'_T.nc' in files:
                print(path+season+"_"+climate_zone+"_"+key+'_T.nc is downloaded')
                counter += 1
                print("Have downloaded in total : " + str(counter) + " files.")
            else:
                c = cdsapi.Client()

                c.retrieve(
                    'reanalysis-era5-single-levels', # reanalysis-era5-pressure-levels
                    {
                        #'class'='ea', # era5 data
                        #'expver' = '1', # ??
                    	'area'    : cz, # retrieving subarea :Europe
                    	"type" : "an", # analysis
                        'product_type':'reanalysis',
                        'format':'grib',
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
                        'variable':'2m_temperature', # , '2m_temperature','surface_pressure','total_cloud_cover'
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
                    path+season+"_"+climate_zone+"_"+key+'_T.grib')
