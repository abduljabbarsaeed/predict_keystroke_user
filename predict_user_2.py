from urllib.request import urlopen
import requests
import json
import pickle
import os
import sys
import pandas as pd

def main(args):

    # storing the JSON response from args.
    data_json = json.loads(args[1])
    print(data_json["Model"])

    # setting REST endpoint url according to "Model".
    if data_json["Model"] == "SVM":
        url = "http://e735e09d-eef9-4c60-87dd-34ffca909437.eastus2.azurecontainer.io/score"   
    elif data_json["Model"] == "RF":
        url = "http://12cddfe9-0c15-4e57-a04e-b1b25a5b71bc.eastus2.azurecontainer.io/score"
    else:
        url = "http://3f09f52c-5823-4e4a-acd1-d3657340ec68.eastus2.azurecontainer.io/score" 

    # setting headers
    headers = {'Content-Type': 'application/json'}

    # removing "Model" key from data_json input
    data_json.pop("Model")
    print(url)
    
    # passing json input in string format to model url and getting response as output. 
    r = requests.post(url,str.encode(json.dumps(data_json)),headers==headers)
    print('predicted user = ', r.json()['user predicted'])

if __name__ == "__main__":
    main(sys.argv)
