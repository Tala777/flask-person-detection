FROM ilyazin/keras-opencv-dlib-tensorflow-api

MAINTAINER Nataliya Kavva <valencia7@ukr.net>

ENV PYTHONPATH=$PYTHONPATH:/opt/models/research:/opt/models/research/object_detection

WORKDIR /opt/project/
COPY requirements.txt /opt/project/requirements.txt
RUN pip3 --no-cache-dir install --upgrade -r requirements.txt
COPY . /opt/project/

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["main.py"]