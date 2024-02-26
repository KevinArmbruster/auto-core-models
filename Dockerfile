FROM msranni/nni

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y apt-transport-https && \
    apt-get install -y git && \
    apt-get install -y curl && \
    apt-get install -y wget && \
    apt-get install -y python3-dev 

WORKDIR /workdir/data
RUN wget https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip
RUN unzip -P someone UCRArchive_2018.zip

WORKDIR /workdir

RUN git clone https://github.com/KevinArmbruster/auto-core-models.git
RUN git config --global user.name "Kevin Armbruster"
RUN git config --global user.email "KevinArmbruster2013@gmail.com"


RUN pip install --upgrade pip setuptools wheel ez_setup
RUN pip install ipython "notebook>=5.3" "ipywidgets>=7.5"
RUN pip install numpy pandas matplotlib scipy scikit-learn rtpt tqdm
RUN pip install seaborn squarify imblearn h5py tables optuna plotly kaleido aeon torchmetrics darts einops
# downgrade protobuf
RUN pip install protobuf==3.20.*


#EXPOSE 8090
ENTRYPOINT ["sleep", "infinity"]

# SETUP INSTRUCTIONS
# mkdir -p ~/auto-core-models/data
# docker build -t auto-core-models-env .
# docker run --gpus all -d auto-core-models-env
#  -v /home/karmbruster/auto-core-models/data:/workdir/datar t
