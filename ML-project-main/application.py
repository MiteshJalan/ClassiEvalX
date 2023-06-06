from flask import Flask,request,render_template,redirect
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline_math import CustomData_math,PredictPipeline_math
from src.pipeline.predict_pipeline_writing import CustomData_writing,PredictPipeline_writing
from src.pipeline.predict_pipeline_reading import CustomData_reading,PredictPipeline_reading

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/to_p1')
def predictdatamath():
    return redirect('/predictdatamath')

@app.route('/to_p2')
def predictdatareading():
    return redirect('/predictdatareading')

@app.route('/to_p3')
def predictdatawriting():
    return redirect('/predictdatawriting')

@app.route('/predictdatamath',methods=['GET','POST'])
def predict_datapoint_math():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData_math(  #object for df datatype for model prediction
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()  #return in df datatype for model prediction
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline_math()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
@app.route('/predictdatareading',methods=['GET','POST'])
def predict_datapoint_reading():
    if request.method=='GET':
        return render_template('home2.html')
    else:
        data=CustomData_reading(  #object for df datatype for model prediction
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=float(request.form.get('math_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df=data.get_data_as_data_frame()  #return in df datatype for model prediction
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline_reading()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home2.html',results=results[0])
    
@app.route('/predictdatawriting',methods=['GET','POST'])
def predict_datapoint_writing():
    if request.method=='GET':
        return render_template('home3.html')
    else:
        data=CustomData_writing(  #object for df datatype for model prediction
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=float(request.form.get('math_score')),
            reading_score=float(request.form.get('reading_score'))
        )

        pred_df=data.get_data_as_data_frame()  #return in df datatype for model prediction
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline_writing()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home3.html',results=results[0])
  

if __name__=="__main__":
    app.debug==True
    app.run(host="0.0.0.0", debug=True)        
