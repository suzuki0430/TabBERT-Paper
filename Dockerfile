# environment.ymlからconda環境を構築(ベースイメージがEC2のAMIと同じ)
FROM 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

COPY environment.yml .

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda init bash && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate tabformer-opt-sagemaker" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV tabformer-opt-sagemaker && \
    PATH /opt/conda/envs/tabformer-opt-sagemaker/bin:$PATH

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/envs/tabformer-opt-sagemaker/lib
RUN echo $LD_LIBRARY_PATH