from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import warnings

warnings.filterwarnings(action='ignore')
app = Flask(__name__)
api = Api(app, version='1.0', title='Diabetes Prediction API', description='API for diabetes prediction')


estimator_rf_model_loaded = joblib.load("saved_models/01.rf_model.pkl")

diabetes_input_model = api.model('DiabetesInput', {
    'Pregnancies': fields.Integer(required=True),
    'Glucose': fields.Integer(required=True),
    'BloodPressure': fields.Integer(required=True),
    'SkinThickness': fields.Integer(required=True),
    'Insulin': fields.Integer(required=True),
    'BMI': fields.Float(required=True),
    'DiabetesPedigreeFunction': fields.Float(required=True),
    'Age': fields.Integer(required=True)
})

diabetes_output_model = api.model('DiabetesOutput', {
    'prediction': fields.Float(description='Predicted diabetes outcome')
})

@api.route('/prediction/diabetes')
class DiabetesPrediction(Resource):
    @api.expect(diabetes_input_model, validate=True)
    @api.marshal_with(diabetes_output_model, code=200)
    def post(self):
        input_data = api.payload

        feature = [[
            input_data['Pregnancies'],
            input_data['Glucose'],
            input_data['BloodPressure'],
            input_data['SkinThickness'],
            input_data['Insulin'],
            input_data['BMI'],
            input_data['DiabetesPedigreeFunction'],
            input_data['Age']
        ]]

        prediction = estimator_rf_model_loaded.predict(feature)[0]

        return {'prediction': prediction}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
