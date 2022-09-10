use role sysadmin;
create database TEST;
use schema TEST.public;
-- create warehouse
CREATE OR REPLACE WAREHOUSE TEST_WH WITH WAREHOUSE_SIZE='XSMALL'
AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
-- create table
create table movies  (
    movieid int,
    title varchar,
    genres varchar
);
create or replace table ratings (
    userid int,
    movieid int,
    rating float,
    timestamp timestamp_ntz
);
-- copy data
put file:///recommendor-system/snowflake-sagemaker-recommendation/ml-latest-small/movies.csv @TEST.PUBLIC.%MOVIES;
copy into TEST.PUBLIC.MOVIES
from @TEST.PUBLIC.%MOVIES
file_format = (type = csv skip_header = 1 field_optionally_enclosed_by='"')
pattern = 'movies.csv.gz'
on_error = 'skip_file';
put file:///recommendor-system/snowflake-sagemaker-recommendation/ml-latest-small/ratings.csv @TEST.PUBLIC.%RATINGS;
copy into TEST.PUBLIC.RATINGS
from @TEST.PUBLIC.%RATINGS
file_format = (type = csv skip_header = 1 field_optionally_enclosed_by='"')
pattern = 'ratings.csv.gz'
on_error = 'skip_file';
-- INPUT TABLE: small dataset for local testing
create table ratings_train_data_local_test as
(select USERID, MOVIEID, RATING from ratings limit 1000);
-- OUTPUT TABLE: top 10 recommendations for local_testing
create or replace table user_movie_recommendations_local_test
(USERID int, TOP_10_RECOMMENDATIONS variant);