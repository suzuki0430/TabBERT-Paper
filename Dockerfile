# environment_sagemaker.ymlからconda環境を構築
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    python3-pip \
    build-essential
WORKDIR /opt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/miniconda3/bin:$PATH

COPY environment_sagemaker.yml .

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -f environment_sagemaker.yml && \
    conda init && \
    echo "conda activate tabformer-opt-sagemaker" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV tabformer-opt-sagemaker && \
    PATH /opt/conda/envs/tabformer-opt-sagemaker/bin:$PATH

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/envs/tabformer-opt-sagemaker/lib
RUN echo $LD_LIBRARY_PATH

WORKDIR /

# CMD ["/bin/bash"]

# # environment.ymlからconda環境を構築
# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# RUN apt-get update && apt-get install -y \
#     sudo \
#     wget \
#     vim \
#     python3-pip
# WORKDIR /opt

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
#     rm -r Miniconda3-latest-Linux-x86_64.sh

# ENV PATH /opt/miniconda3/bin:$PATH

# COPY environment.yml .

# RUN pip install --upgrade pip && \
#     conda update -n base -c defaults conda && \
#     conda env create -f environment.yml && \
#     conda init && \
#     echo "conda activate tabformer-opt" >> ~/.bashrc

# ENV CONDA_DEFAULT_ENV tabformer-opt && \
#     PATH /opt/conda/envs/tabformer-opt/bin:$PATH

# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/envs/tabformer-opt/lib
# RUN echo $LD_LIBRARY_PATH

# WORKDIR /

# CMD ["/bin/bash"]

# FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

# RUN apt-get update && apt-get install -y \
#     sudo \
#     wget \
#     vim
# WORKDIR /opt

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
#     rm -r Miniconda3-latest-Linux-x86_64.sh

# ENV PATH /opt/miniconda3/bin:$PATH

# COPY environment.yml .

# RUN pip install --upgrade pip && \
#     conda update -n base -c defaults conda && \
#     conda env create -n tabformer -f environment.yml && \
#     conda init && \
#     echo "conda activate tabformer" >> ~/.bashrc

# ENV CONDA_DEFAULT_ENV <env_name> && \
#     PATH /opt/conda/envs/tabformer/bin:$PATH

# WORKDIR /

# CMD ["/bin/bash"]


# FROM nvidia/cuda:11.0.3-devel-ubuntu16.04

# RUN apt-get update && apt-get install -y \
#     sudo \
#     wget \
#     vim \
#     python3-pip
# WORKDIR /opt

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
#     rm -r Miniconda3-latest-Linux-x86_64.sh

# ENV PATH /opt/miniconda3/bin:$PATH

# COPY setup.yml .

# RUN pip install --upgrade pip && \
#     pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && \
#     conda update -n base -c defaults conda && \
#     conda config --set channel_priority flexible && \
#     conda env create -f setup.yml && \
#     conda init && \
#     echo "conda activate tabformer" >> ~/.bashrc

# ENV CONDA_DEFAULT_ENV tabformer && \
#     PATH /opt/conda/envs/tabformer/bin:$PATH

# WORKDIR /

# CMD ["/bin/bash"]
