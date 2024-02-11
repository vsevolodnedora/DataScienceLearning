__Course__:
- Create Model API
- Create Docker Image
- Build Container - Deply

__Model__: language detection model: output is dependent on the language od the requries

- imput -> Docker container -> recieve responds in json (as output)

- Docker creates a _virtualization layer_ on top of the operation system that allows to run applications.  
Software there does not allow any extra installations

Docker takes:
- code
- libraries
- data base 
and bundles it into a container that runs on aly system

docker allows to expand the work , build on containers

Docker is a _blueprint_ in a Dockerfile, which is a step-by-step gide. this image is used to compile an _image_. Image is a static object that is not modified. It can be used to start a container that runs in its own environemnt, independently of the opeartion system 

An image can be pushed to a _Registry/Hub_ that contains all images and all _images_. Than this image can be download. Inside there is a _compression/decompression_. 

Docker is not a virtual machine, where an VM runs a full version and can run everything. Docker bundles only the _important_ logic that allows to save space and run faster. 

__NOTE__ you cannot run docker in docker. You cannot run it there. 

---  
__Example__:  
1. (Optional) Look for the ```hello-world``` Docker image on Docker hub: https://hub.docker.com/
2. Download the Docker image using the command: ```docker pull hello-world```
3. List your Docker images with the command: ```docker image ls```
4. Create a container using the ```hello-world``` image by running: ```docker run hello-world```
5. List your Docker containers using: ```docker ps --all```
6. Remove your Docker container with the command: ```docker rm [ID]```
7. Create a self-deleting container with the ```hello-world``` image using: ```docker run --rm hello-world```
do
```bash
docker run -it ubuntu:22.04 bash # run a container and install bash concil
```

# Building an Application

- Make a dataset
- Train a model
- Dowload a model

### Input:
- Ciao -> scikit-learn -> Italian

Use Jupyter to run a container. 

You can use Docker to spawn a Jypiter

Run the command exposing the port: 

1. Download the image: ```docker pull jupyter/base-notebook```
2. Start a self-deleting container using the command: ```docker run --rm jupyter/base-notebook```
3. Read the terminal output can you access it?
5. Forward the port ```8888``` from the Docker container to the host machine: ```docker run -p 8888:8888 --rm jupyter/base-notebook```

- Docker container is empty. It has an empty _home_ direcotry.  

To solve it we need to expose the host machine storage to container. 
This is done with _mounting_. 
- Docker _Volume Mount_ - the most recommended way. Storage is independent of whether the container is up or not. 
- _Bind Mount_ 
    - Real time code changes
    - access o Sytem files
    - Configuration Managment
- tmpfs Mount 
    - Non-persistent data
    - fast data processing
    - in-memory cashing

onsider the _Bound Mount_
- craete a specific _docker area_ insider the filesstem 

--- 
**Problem:** The container cannot access our file on our host machine. 

**Solution:** Use a bind mount to access our **model** directory inside the Jupyter Notebook.

**Task:** Complete the following command: ```docker run […] -p 8888:8888 --rm jupyter/base-notebook```

1. Read Docker Documentation: https://docs.docker.com/storage/bind-mounts/
2. **Hint:**  Mount the **model** source folder from the clone of the workshop code to the ```/home/jovyan/work``` target folder in the Docker container.


# Craeting an API

Client -> Requiest -> API -> Server -> API -> Client

### API calls with FastAPI

__API methods__:
- POST (submit input features to model)
- GET (most used) Retrieve info; 
- PUT
- DELETE

```python
app = FastAPI()
@app.get("/")
async def root():
    return {"message"}
```

FastAPI = WebServer that can be accessed via browser

```python
from fastapi import FastAPI, Form, Depends
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
from typing import Tuple
import sklearn
import numpy as np

model_api = FastAPI()

class Result(BaseModel):
    language : str # using the base class __init__ method

def get_model() -> Tuple[sklearn.base.BaseEstimator, np.matrix]:
    full_model = joblib.load("model/multinomial_language_detector.joblib")
    return full_model

# TODO: Create a post request with the path “/predict" and the Result response_model.
@model_api.post("/predict", # end point is predict
                response_model=Result)
async def predict(input_text: str = Form(), 
                 full_model : Tuple = Depends(get_model)) -> Result:
    model, cv = full_model
    vectorized = cv.transform([input_text])
    language_prediction = model.predict(vectorized)[0]
    # TODO: Return a Result Object with the language being the predicted language.
    return Result(language_prediction)

model_api.mount("/", StaticFiles(directory="app/static", html=True), name="static")
if __name__ == "__main__":
    uvicorn.run(model_api, host="0.0.0.0")

```

# Containerizing the API

- FROM specifies the base image like _python:3.11_ which is an operation system with pyhthon in it 
- WORKDIR - sets a working and sets it as a default path (cd)
- COPY - copy files into image (copy e.g., an entire application)
- RUN - run e.g. pip install
- CMD - Used only once per docker file -- starts the server with 

Each of these is a _layer_ in the docker file. 

After, use _docker build_ to actually launch

```Dockerfile
# use python image as a parent image
FROM python:3.11-slim
# set the working dir in the ocntainer
WORKDIR /app
# copy dependencies file
COPY ./requirements-prod.txt ./
# install any needed packages 
RUN pip install -r requirements-prod.txt
# copy the rest of the application
COPY ./ ./
# run the command to start the uvicorn
CMD ["uvicorn","app.api:model_api","--host","0.0.0.0"]
```

then run to actually build the image

```bash
docker compose up --build
```