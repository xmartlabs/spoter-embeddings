FROM pytorch/pytorch

WORKDIR /app
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app/
WORKDIR /app/
CMD ./train.sh
