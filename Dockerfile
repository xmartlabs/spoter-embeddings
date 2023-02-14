FROM pytorch/pytorch

WORKDIR /app
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app/
RUN git config --global --add safe.directory /app
CMD ./train.sh
