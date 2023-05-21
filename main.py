from fastapi import FastAPI

from api.models.sample import Sample
from api.models.irisParams import IrisParams
from dotenv import load_dotenv

import api.services.hello as hello_service
import api.services.kedro as kedro_service
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import os
import json

load_dotenv()

app = FastAPI()

@app.post("/hello")
async def create_hello(sample: Sample):
    res = hello_service.create_hello(sample)
    return res

@app.get("/hello")
async def get_hello():
    res = hello_service.get_hello()
    return res

@app.post("/kedro")
async def run_simple_analysis(sample: Sample):
    res = kedro_service.run(sample)
    return res

@app.post("/iris")
async def run_iris_analysis(irisParams: IrisParams):
    print(os.getenv("ANALYTICS_MODULE_PATH"))
    bootstrap_project(os.path.abspath(os.getenv("ANALYTICS_MODULE_PATH")))
    os.chdir(os.getenv("ANALYTICS_MODULE_PATH"))
    print(irisParams)
    with KedroSession.create(extra_params=irisParams.__dict__) as session:
        res = session.run(pipeline_name="iris")
        return res

@app.get("/games")
async def run_pandas_analysis():
    print(os.getenv("ANALYTICS_MODULE_PATH"))
    bootstrap_project(os.path.abspath(os.getenv("ANALYTICS_MODULE_PATH")))
    os.chdir(os.getenv("ANALYTICS_MODULE_PATH"))
    with KedroSession.create() as session:
        res = session.run(pipeline_name="games")
        return {
            "result": json.loads(res["games"])
        }

if __name__ == "__main__":
    print("Server runner: Uvicorn")
