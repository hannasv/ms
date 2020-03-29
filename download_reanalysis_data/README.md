# Scripts for downloading reanalysis data
Both era-interim and ERA5 was considered for this project, scripts for downloading era
interim can be found in directory **downloading_era_interim** and ERA5 in **download_ERA5**. 

## Instructions for downloading ERA5 using the CDS-api
Inspired by ECMWF's own instructions found at https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets.
First creata a user at CDS registration. Go to https://cds.climate.copernicus.eu/#!/home}{https://cds.climate.copernicus.eu/#!/home and create a user. Log in to retrieve you key. This will be valid for 12 months. Paste the key in a document and save it as **\$HOME/.cdsapirc** on a Unix/Linux platform. 

```bash
url: {api-url}
key: {uid}:{api-key}
```
The MARS downloading efficiency limits you to 100 000 items in one request. Workaround is looping over the requests.

ECMWFs API generator for surface parameters (temperature, pressure, total cloud cover) https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form.

ECMWF's API for pressure level parameters: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form.

Note that neither of these include area and type, so these need to be added. Other than that the above links are useful for generating the API request necessary.

This is available throught the project enviornment _sciclouds_ (installing inststructions are available in root). 
```python 
pip install cdsapi # installing package
python download_file.py # executing script
``` 
## Example code 

```python 
import cdsapi

c = cdsapi.Client()

c.retrieve('reanalysis-era5-pressure-levels',
    {
    	'area'    : "75/-15/30/42", # retrieving subarea :Europe
    	"type" : "an", # analysis
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
        'month':['01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'],
        'year': ['1979', '1980'],
        'pressure_level':[
            '250','600','850',
            '1000'
        ],
        'variable':'relative_humidity',
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
    'rh.nc')
``` 
