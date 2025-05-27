FROM tensorflow/tensorflow:latest-gpu-jupyter

# Installiere zusätzliche Python-Bibliotheken
RUN pip install --no-cache-dir scikit-learn pandas matplotlib seaborn torch 

WORKDIR /workfiles

# Expose Jupyter Notebook?
#EXPOSE 8888