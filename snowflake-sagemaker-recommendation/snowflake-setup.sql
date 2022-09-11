/* 1. Snowflake에 샘플 데이터 로드 */
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
-- load data
put file:///recommendor-system/snowflake-sagemaker-recommendation/ml-latest-small/movies.csv @TEST.PUBLIC.%MOVIES;
/*
+------------+---------------+-------------+-------------+--------------------+--------------------+----------+---------+
| source     | target        | source_size | target_size | source_compression | target_compression | status   | message |
|------------+---------------+-------------+-------------+--------------------+--------------------+----------+---------|
| movies.csv | movies.csv.gz |      494431 |      166576 | NONE               | GZIP               | UPLOADED |         |
+------------+---------------+-------------+-------------+--------------------+--------------------+----------+---------+
1 Row(s) produced. Time Elapsed: 4.875
*/
copy into TEST.PUBLIC.MOVIES
from @TEST.PUBLIC.%MOVIES
file_format = (type = csv skip_header = 1 field_optionally_enclosed_by='"')
pattern = 'movies.csv.gz'
on_error = 'skip_file';
/*
+---------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------+
| file          | status | rows_parsed | rows_loaded | error_limit | errors_seen | first_error | first_error_line | first_error_character | first_error_column_name |
|---------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------|
| movies.csv.gz | LOADED |        9742 |        9742 |           1 |           0 | NULL        |             NULL |                  NULL | NULL                    |
+---------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------+
1 Row(s) produced. Time Elapsed: 1.281s
*/
put file:///recommendor-system/snowflake-sagemaker-recommendation/ml-latest-small/ratings.csv @TEST.PUBLIC.%RATINGS;
copy into TEST.PUBLIC.RATINGS
from @TEST.PUBLIC.%RATINGS
file_format = (type = csv skip_header = 1 field_optionally_enclosed_by='"')
pattern = 'ratings.csv.gz'
on_error = 'skip_file';

/* 2. 로컬 테스트를 위해 테스트 테이블 생성 */
-- INPUT TABLE: small dataset for local testing
create table ratings_train_data_local_test as
(select USERID, MOVIEID, RATING from ratings limit 1000);
-- OUTPUT TABLE: top 10 recommendations for local_testing
create or replace table user_movie_recommendations_local_test
(USERID int, TOP_10_RECOMMENDATIONS variant);

/* 9. API Gateway를 Snowflake와 연결한다. */
use role accountadmin;
use schema TEST.PUBLIC;
create or replace api integration snf_recommender_api_integration
api_provider = aws_api_gateway
api_aws_role_arn = 'arn:aws:iam::123456789012:role/SnowflakeAPIGatewayTest'
enabled = true
api_allowed_prefixes = ('https://xxxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/');

/* 10. Snowflake에 External functions을 생성한다. */
-- create external function : train_and_get_recommendations
create or replace external function train_and_get_recommendations(input_table_name varchar, output_table_name varchar)
    returns variant
    api_integration = snf_recommender_api_integration
    as 'https://xxxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/train';
-- create external function : deploy_model
create or replace external function deploy_model(model_name varchar, model_url varchar)
returns variant
api_integration = snf_recommender_api_integration
as 'https://xxxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/deploy';
-- create external function : invoke_model
create or replace external function invoke_model(model_name varchar, user_id varchar, item_id varchar)
returns variant
api_integration = snf_recommender_api_integration
as 'https://xxxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/invoke';
-- grant permissions to sysadmin
grant usage on function train_and_get_recommendations(varchar, varchar) to role sysadmin;
grant usage on function deploy_model(varchar, varchar) to role sysadmin;
grant usage on function invoke_model(varchar, varchar, varchar) to role sysadmin;

/* 11. External functions 테스트를 위한 테이블을 생성한다. */
-- create test table
create or replace table ratings_train_data as 
    (select USERID, MOVIEID, RATING 
    from ratings limit 10000);
create or replace table user_movie_recommendations 
(USERID float, 
TOP_10_RECOMMENDATIONS variant);