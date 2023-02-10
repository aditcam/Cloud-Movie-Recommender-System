## Cloud-Movie-Recommender-System
Deployed a Pyspark based movie recommendation solution on GCP using Collaborative Filtering (Alternating Least Squares algorithm) in a Dataproc environment.

Tech Stack : PySpark + Python + GCP (Dataproc, Jupyter, Cloud Storage Buckets, VPC networks, BigQuery, Cloud Shell)

Leveraged the Google Cloud Platform to implement a Movie Recommender system for a given user using PySpark. 

In addition to the recommender system, some data transformations, computation of summary stats and visualizations were also created in a jupyter notebook on the Dataproc cluster.

The dataset used is the movielens dataset by Grouplens, the MovieLens 1M Dataset in particular which is a stable benchmark dataset consisting of 1 million movie ratings from 6000 users on 4000 movies, it was released in 2003.

<img width="1104" alt="image" src="https://user-images.githubusercontent.com/22599347/218106304-c9b17565-dae8-4ec9-8b03-ed034395e73a.png">

The flow of the program is as shown above – 

1) The first step is to ingest the dataset from the external source i.e. Movielens in this case. I accomplished this using cloud shell as my staging area where I used the dataset download url and ran a curl command to download the zip file. The next this is to unzip the files in cloud shell and then move it to a google cloud storage bucket using a gsutil command. With this we have successfully ingested the dataset. 

2) The next step to accomplish is to then load the datasets from the gcp buckets and perform some analysis on them. I do this by creating a Jupyter notebook instance in my dataproc cluster. In my analysis I initially transform the data in order to make analysis easier, then create some summary statistics on the dataset and finally produce visualizations using matplotlib that help us understand the dataset better.

3) Once the preliminary analysis is done, I then run a spark job on my dataproc cluster which accomplishes 2 things, first one is to generate the top n movie recommendations (I have chosen n = 3) for every user in the dataset. The second thing this spark job does is migrate the all the data ( The original dataset along with the predictions ) into a Bigquery dataset in the form of tables in order to create a data warehouse.

4) Finally we can navigate to big query and query the dataset based on our requirements, create views and perform other database operations.

**NOTE** : The visualizations are present in the Jupyter notebook (.ipynb file). The recommender.py script was the driver program for the dataproc job used in GCP.
The pdf file consists of detailed explanations and how to replicate this project step by step. 
