# Lab Walkthrough Video

End-to-end data pipeline on AWS
- Source: relational DB
- Ingestion to pipeline
- transport an load 
- serve

Transformation
- intransform into a more managable format

Main goal: learn to interact with data pripleine

Business: Retail  
Branch: Cars  
__Data:__ relational database of historical purchases and customer's info
Goal: transform the data to be used by data analyst that wants to find out
- Which product lines are more successfull
- How sales are distributed across different countries

Workflow: extract data from DB, transform, save to a place that an analyst has access to

> __Star Schema__ - desired data format where there is a main table (__fact table__) with key measurements and ancillary tables (__dimensions__) with more context

Workflow: Given a source database, setup a _glue job_ that 
- _Extracts_ data from RDS database
- _Transforms_ data using the stak schema
- _Loads_ data into S3 bucket

Last two stages are called (__ETL__)

We will use _Glue crawler_ to crawl over S3 and write metadata to a data catalogue

Data can be retrieved from S3 using SQL-like queries

In this lag, the ETL will be setup programmatarically using __Terraform__.



# An Example of the Data Engineering Lifecycle


