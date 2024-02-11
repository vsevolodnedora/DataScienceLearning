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