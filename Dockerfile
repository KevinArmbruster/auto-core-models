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

WORKDIR /workdir


# nvm install, load
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# RUN export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")" [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"


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
ENTRYPOINT ["tail", "-f", "/dev/null"]

# SETUP INSTRUCTIONS
# mkdir -p ~/auto-core-models/data
# docker build -t auto-core-models-env .
# docker run -d auto-core-models-env
#  -v /home/karmbruster/auto-core-models/data:/workdir/datar t