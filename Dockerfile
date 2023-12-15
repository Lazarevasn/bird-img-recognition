FROM python:3.8.10

WORKDIR /workspace

RUN python -m pip install --upgrade pip
RUN pip install jupyterlab

COPY requirements.txt /workspace

RUN pip install --no-cache -r requirements.txt

COPY . /workspace
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]