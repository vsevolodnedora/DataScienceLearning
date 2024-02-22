# Docker workshop at HPI on 06.02.2024

Workshop: “Containerize your Intelligence”
On February 06, 2024, we offer the free workshop “Containerize your Intelligence” on site at HPI.

For this hands-on Workshop on the use of AI models with Docker, no previous knowledge of FastAPI or Docker is required, therefore it is ideal for all AI enthusiasts, data scientists and developers.


Imagine you have a trained and fine-tuned AI model, but don't know how to use it in practice. Our workshop will help you to transform your model into an operational AI solution. 

Through practical exercises, we will show you the path from model training to deployment. You will learn the basics of deploying AI models and gain insights into containerization with Docker and the creation of AI interfaces with FastAPI. 

    1  docker
    2  ls
    3  whoemi
    4  whoami
    5  docker pull hello-world
    6  docker image ls
    7  docker run hellow-world
    8  docker run hello-world
    9  docker ps --all
   10  docker rm hello-world
   11  docker rm hellow-world
   12  docker rm hello-world
   13  docker rm a70dbe55a2bf
   14  docker logs a70dbe55a2bf
   15  docker pull jupyter/base-notebook
   16  docker run --rm jupyter/base-notebook
   17  docker run -p 8888:8888 --rm jupyter/base-notebook
   18  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source="$(pwd)"/model,target=/model
   19  ls
   20  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source="$(pwd)"/model,target=/home/jovian
   21  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source="$(pwd)"/model,target=/home/jovyan
   22  pwd
   23  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source=./model,target=/home/jovyan
   24  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source=~/model,target=/home/jovyan
   25  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source=~/model,target=/home/jovian/model
   26  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source="$(pwd)"/model,target=/home/jovyan
   27  docker run -p 8888:8888 --rm jupyter/base-notebook --mount type=bind,source="$(pwd)"/model,target=/home/jovyan/
   28  docker run -p 8888:8888 --mount type=bind,source="$(pwd)"/model,target=/home/jovyan/ --rm jupyter/base-notebook
   29  docker run -p 8888:8888 --mount type=bind,source="$(pwd)"/model,target=/home/jovyan/work --rm jupyter/base-notebook
   30  git checkout solutions -- model/
   31  git checkout solutions --model/
   32  git checkout solutions -- model/
   33  ls
   34  python3 app/api.py 
   35  clear
   36  ls
   37  docker compose up --build
   38  docker image ls
   39  git status
   40  git add Dockerfile 
   41  git add app/api.py 
   42  git add model/model_trainer.ipynb 
   43  git commit -m "my solution"
   44  git pull
   45  git push
   46  hsitory
   47  history
   48  history -> history.txt
   49  history > history.txt