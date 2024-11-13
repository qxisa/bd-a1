#using ubuntu 20.04 as base image
FROM ubuntu:20.04 


#installing python3, python3-pip, python3-dev, pandas, numpy, seaborn, matplotlib, scikit-learn, scipy, docker.io as instructred in pdf
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip \
    && pip3 install pandas numpy seaborn matplotlib scikit-learn scipy \
    && apt-get install -y docker.io


#making a directory in the container 
RUN mkdir -p /home/doc-bd-a1/

#copying the dataset to the container
COPY trainDS.csv /home/doc-bd-a1/

#working directory is set to the directory created
WORKDIR /home/doc-bd-a1/


#cmd to run the python script using python3, the script is load.py and the dataset is trainDS.csv and the code
CMD ["python3", "load.py", "/home/doc-bd-a1/trainDS.csv"]
