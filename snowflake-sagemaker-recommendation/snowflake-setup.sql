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
-- connect API Gateway with Snowflake
use role accountadmin;
use schema TEST.PUBLIC;
create or replace api integration snf_recommender_api_integration
api_provider = aws_api_gateway
api_aws_role_arn = 'arn:aws:iam::717473574740:role/SnowflakeAPIGatewayTest'
enabled = true
api_allowed_prefixes = ('https://pgvn9t3vhc.execute-api.us-east-1.amazonaws.com/dev/');
-- create external functions
create or replace external function train_and_get_recommendations(input_table_name varchar, output_table_name varchar)
    returns variant
    api_integration = snf_recommender_api_integration
    as 'https://pgvn9t3vhc.execute-api.us-east-1.amazonaws.com/dev/train';
create or replace external function deploy_model(model_name varchar, model_url varchar)
returns variant
api_integration = snf_recommender_api_integration
as 'https://pgvn9t3vhc.execute-api.us-east-1.amazonaws.com/dev/deploy';
create or replace external function invoke_model(model_name varchar, user_id varchar, item_id varchar)
returns variant
api_integration = snf_recommender_api_integration
as 'https://pgvn9t3vhc.execute-api.us-east-1.amazonaws.com/dev/invoke';
-- grant permissions to sysadmin
grant usage on function train_and_get_recommendations(varchar, varchar) to role sysadmin;
grant usage on function deploy_model(varchar, varchar) to role sysadmin;
grant usage on function invoke_model(varchar, varchar, varchar) to role sysadmin;