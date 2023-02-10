from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import explode
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
import time

spark = SparkSession.builder.appName('Spark_ALS_recommender').getOrCreate()

user_col = ['user_id', 'gender', 'age', 'occupation', 'zip']
user_df = spark.read.format("csv").option("delimiter", "::").option("inferSchema", "true").load("gs://adcama-movie-recommender-data/data/users.dat")
col_alias = user_df.columns
user_df = user_df.select([col(col_alias[i]).alias(user_col[i]) for i in range(len(user_col))])

rating_col = ['user_id', 'movie_id', 'rating', 'timestamp']
rating_df = spark.read.format("csv").option("delimiter", "::").option("inferSchema", "true").load("gs://adcama-movie-recommender-data/data/ratings.dat")
rating_df = rating_df.select([col(col_alias[i]).alias(rating_col[i]) for i in range(len(rating_col))])

movie_col = ['movie_id', 'title', 'genres']
movie_df = spark.read.format("csv").option("delimiter", "::").option("inferSchema", "true").load("gs://adcama-movie-recommender-data/data/movies.dat")
movie_df = movie_df.select([col(col_alias[i]).alias(movie_col[i]) for i in range(len(movie_col))])

(train, test) = rating_df.randomSplit([0.8, 0.2])

als = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating",coldStartStrategy="drop",nonnegative=True,implicitPrefs=False)

# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10,100,150]) \
            .addGrid(als.regParam, [.1, .15,.3]) \
            .addGrid(als.maxIter, [1, 5]) \
            .build()

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
print ("Total number of models to be tested: ", len(param_grid))

# Build cross validation using CrossValidator
start = time.time()
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)
best_model = cvModel.bestModel
end = time.time()
print('Time taken for training {}s'.format(round(end-start,2)))

print("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

# View the predictions
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print('Root Mean Squared Error is {}'.format(RMSE))

movieSubSetRecs = best_model.recommendForAllUsers(3)

movieSubSetRecs.show(10,False)

final_df = movieSubSetRecs \
.withColumn('recommendation',explode(movieSubSetRecs.recommendations)) \
.select(movieSubSetRecs.user_id,col('recommendation.movie_id'),col('recommendation.rating'))

print('Moving tables to bigquery .... ')

final_df.write.format('bigquery') \
  .option("writeMethod", "direct") \
  .mode("overwrite") \
  .save('movielens_recommendation.user_movie_recs')

user_df.write.format('bigquery') \
  .option("writeMethod", "direct") \
  .mode("overwrite") \
  .save('movielens_recommendation.users')

movie_df.write.format('bigquery') \
  .option("writeMethod", "direct") \
  .mode("overwrite") \
  .save('movielens_recommendation.movies')

rating_df.write.format('bigquery') \
  .option("writeMethod", "direct") \
  .mode("overwrite") \
  .save('movielens_recommendation.ratings')

