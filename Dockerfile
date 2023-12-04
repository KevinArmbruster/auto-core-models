FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y apt-transport-https && \
    apt-get install -y git && \
    apt-get install -y curl && \
    apt-get install -y wget

WORKDIR /workdir

RUN git clone https://github.com/KevinArmbruster/auto-core-models.git
RUN git config --global user.name "Kevin Armbruster"
RUN git config --global user.email "KevinArmbruster2013@gmail.com"


RUN pip install --upgrade pip
RUN pip install ipython "notebook>=5.3" "ipywidgets>=7.5"
RUN pip install numpy pandas matplotlib scipy scikit-learn
RUN pip install h5py tables
RUN pip install rtpt tqdm
RUN pip install seaborn squarify imblearn
RUN pip install optuna plotly kaleido
RUN pip install aeon torchmetrics darts einops


#EXPOSE 8090
ENTRYPOINT ["tail", "-f", "/dev/null"]

# SETUP INSTRUCTIONS
# mkdir -p ~/auto-core-models/data
# docker build -t auto-core-models-env .
# docker run -v /home/karmbruster/auto-core-models/data:/workdir/data -d auto-core-models-env