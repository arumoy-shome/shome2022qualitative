# general setup
FROM python:3.9
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# setup the datasets we will use in our analysis
COPY datasets.zip ./
RUN unzip datasets.zip
RUN cp datasets/adult.* /usr/local/lib/python3.9/site-packages/aif360/data/raw/adult/
RUN cp datasets/german.data /usr/local/lib/python3.9/site-packages/aif360/data/raw/german/
RUN cp datasets/compas-* /usr/local/lib/python3.9/site-packages/aif360/data/raw/compas/
RUN cp datasets/bank-* /usr/local/lib/python3.9/site-packages/aif360/data/raw/bank/
RUN cp datasets/h1*.csv /usr/local/lib/python3.9/site-packages/aif360/data/raw/meps/

