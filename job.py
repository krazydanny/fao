import os
import pickle
import shutil
import zipfile
import datetime
import pandas as pd
import urllib.request

from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

e 						= 0
storage 				= './data'
modelFileName			= './emissions_model.pkl'
scalerFileName			= './emissions_scaler.pkl'
dataFileName			= './emissions_data.csv'

# download datasets from FAO.org
livestockProductionUrl  = 'http://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Livestock_E_All_Data_(Normalized).zip'
agricultureEmissionsUrl = 'http://fenixservices.fao.org/faostat/static/bulkdownloads/Emissions_Agriculture_Agriculture_total_E_All_Data_(Normalized).zip'
filesEncoding 			= 'ISO-8859-1'


try:
	shutil.rmtree( storage )
except OSError as e:
	e = 1


try:
	os.mkdir( storage , 0o755 )
except OSError as e:
	e = 1


urllib.request.urlretrieve(
	livestockProductionUrl, 
	storage + '/livestockProduction.zip'
)

urllib.request.urlretrieve(
	agricultureEmissionsUrl, 
	storage + '/agricultureEmissions.zip'
)


# unzip the files in order to make .csv files available
with zipfile.ZipFile( storage + '/livestockProduction.zip', 'r' ) as zip_ref:
	zip_ref.extractall( storage )

with zipfile.ZipFile( storage + '/agricultureEmissions.zip', 'r' ) as zip_ref:
	zip_ref.extractall( storage )


# load, group and merge datasets
emissionsTotal = pd.read_csv(
	storage + '/Emissions_Agriculture_Agriculture_total_E_All_Data_(Normalized).csv', 
	header = 0, 
	sep = ",", 
	encoding = filesEncoding
)

emissionsTotalGrouped = emissionsTotal[ (emissionsTotal['Element']=='Emissions (CO2eq) from CH4') & (emissionsTotal['Year']<datetime.datetime.now().year) ].groupby(['Area Code', 'Area', 'Year Code']).sum().reset_index().drop(columns=['Year', 'Element Code', 'Item Code', 'Note']).rename(columns={"Value": "emissions_gigagrams","Area Code": "area_code", "Area": "area", "Year Code": "year"})


livestockProduction = pd.read_csv(
	storage + '/Production_Livestock_E_All_Data_(Normalized).csv', 
	header = 0, 
	sep = ",", 
	encoding = filesEncoding
)

livestockProductionGrouped = livestockProduction.groupby(['Area Code', 'Area', 'Year Code']).sum().reset_index().drop(columns=['Year', 'Element Code', 'Item Code']).rename(columns={"Value": "animals_stock", "Area Code": "area_code", "Area": "area", "Year Code": "year"})


data = pd.merge( emissionsTotalGrouped, livestockProductionGrouped.drop( columns=['area'] ), how="left", left_on=['area_code', 'year'], right_on=['area_code', 'year'] )

# remove rows without animals_stock values
data = data[ ~data['animals_stock'].isnull() ]

try:
	os.remove( dataFileName )
except OSError as e:
	e = 1

data.to_csv( dataFileName )

# exclude areas which duplicate data
excludedAreas = [
        'World', 'Africa', 'China, mainland',
       'Eastern Africa', 'Middle Africa', 'Northern Africa',
       'Southern Africa', 'Western Africa', 'Americas',
       'Northern America', 'Central America', 'Caribbean',
       'South America', 'Asia', 'Central Asia', 'Eastern Asia',
       'Southern Asia', 'South-Eastern Asia', 'Western Asia', 'Europe',
       'Eastern Europe', 'Northern Europe', 'Southern Europe',
       'Western Europe', 'Oceania', 'Australia & New Zealand',
       'Melanesia', 'Micronesia', 'Polynesia', 'European Union',
       'Least Developed Countries', 'Land Locked Developing Countries',
       'Small Island Developing States',
       'Low Income Food Deficit Countries',
       'Net Food Importing Developing Countries','Annex I countries',
       'Non-Annex I countries', 'OECD'
]

areaCodes  = emissionsTotal[ ~emissionsTotal.Area.isin(excludedAreas) ]['Area Code'].unique()
productionAreaCodes = livestockProduction[ ~livestockProduction.Area.isin(excludedAreas) ]['Area Code'].unique()

areaCodesDiff = list( set(productionAreaCodes) - set(areaCodes) )

for n in areaCodesDiff:
    areaCodes.append(n)


# select x
x = data[ data['area_code'].isin(areaCodes) ].drop( columns=['emissions_gigagrams', 'area_code'])


# use one-hot encoding for categorical features
x = pd.concat( [ x, pd.get_dummies(x['area'], prefix='country') ], axis=1 )
x.drop( columns=['area'], inplace=True )


# select y
y = data[ data['area_code'].isin(areaCodes) ]['emissions_gigagrams']


# split in train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=41)

# standarize
scaler  = StandardScaler()

x_train = scaler.fit_transform( x_train )
x_test  = scaler.transform( x_test )

pickle.dump( scaler, open( scalerFileName, 'wb' ) )

# model train with gridsearch and save

model = LassoCV( 
	cv = 3,
	n_jobs = -1,
	random_state = 42
)

model.fit( x_train, y_train )

try:
	os.remove( modelFileName )
except OSError as e:
	e = 1

pickle.dump( model, open( modelFileName, 'wb' ) )