import pickle
import jsonschema
import pandas as pd
from flask import Flask, request, abort, jsonify
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

filesEncoding  = 'ISO-8859-1'
modelFileName  = './emissions_model.pkl'
scalerFileName = './emissions_scaler.pkl'
dataFileName   = './emissions_data.csv'

with open('./emissions_model.pkl', 'rb') as pickleFile:
    model = pickle.load(pickleFile)

with open('./emissions_model.pkl', 'rb') as pickleFile:
    model = pickle.load(pickleFile)    

app = Flask(__name__)

@app.route('/')
def index():
  return 'Welcome!'
  
@app.route('/predict', methods=["POST"])
def predict():

	# load train dataset to get columns
	data = pd.read_csv(
		'./data.csv', 
		header = 0, 
		sep = ",", 
		encoding = filesEncoding
	)

	# validate request format
	schema = {
		'type': 'array',
		'items': {
			'type': 'object',
			'properties': {
				'animals_stock': { 
					'type': 'integer',
				},
				'year': { 
					'type': 'integer',
				},
				'country': { 
					'type': 'string',
				},
			},
			'required': [ 'animals_stock' ],		
		}
	}

	inputs = request.json

	try:
		jsonschema.validate(
			instance=inputs, 
			schema=schema
		)

		# create empty dataframe with all columns
		x = pd.DataFrame( columns = data.columns )

		# fill with input
		x.append( inputs )

		#x = pd.concat( [ x, pd.get_dummies(x['country'], prefix='country') ], axis=1 )
		#x.drop( columns=['country'], inplace=True )

		# standarize
		x  	   = scaler.transform( x )

		y_predicted = model.predict( x )

		return jsonify( y_predicted )

	except jsonschema.exceptions.ValidationError as e:
		abort(400)

if __name__ == '__main__':
    app.run(debug=True)