FROM continuumio/miniconda3

# save environment.yml
ADD environment.yml /tmp/environment.yml

# create conda environment
RUN conda env create -f /tmp/environment.yml

# add command to activate conda environment
RUN echo "conda activate myenv" >> ~/.bashrc

# add files to the venv
ADD . .