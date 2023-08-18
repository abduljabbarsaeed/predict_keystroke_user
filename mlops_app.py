from flask import Flask, request
import subprocess
import json

app = Flask(__name__)

in_memory_datastore = [
    {
        "Model": "RF",
        "HT": {"Mean": 48.43,"STD": 23.34},
        "PPT": {"Mean": 120.43,"STD": 37.41},
        "RRT": {"Mean": 124.43,"STD": 45.34},
        "RPT": {"Mean": 132.56,"STD": 47.12}
    }
]

@app.get('/model_inputs')
def list_model_inputs():
    return in_memory_datastore

@app.route('/model_inputs', methods=['GET', 'POST'])
def programming_languages_route():
   if request.method == 'GET':
       return list_model_inputs()
   elif request.method == "POST":
       return create_model_inputs(request.get_json(force=True))

def create_model_inputs(model_inputs):
   in_memory_datastore[0] = json.dumps(model_inputs)
   #print(in_memory_datastore[0])
   user = subprocess.check_output(["python3","predict_user_1.py",str(in_memory_datastore[0])])
   user = user.decode("utf-8")
   user = int(user.replace("\n",""))
   #print(user)
   model_inputs["user"] = user
   in_memory_datastore[0] = model_inputs
   return in_memory_datastore
