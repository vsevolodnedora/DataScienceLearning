# Data Generation in Source Systems

__Examples__:
- Sales transaction data from internal database
- Social Media data from APIs
- Data from IOTs 

Maintained by software engineers



__Commons sourcers__:
- Databases
    - Relational, SQL 
    - NoSQL databases (document stores)
- Files
    - Text, audio, video
- APIs (application Programming Interface)
- Data Sharing Platform
- IOT devices (individual devices, streaming data in real time, accessible via APIs)

Source systems are commonly _upredictable_ and schemas and foremats may change. 

In order to build reliable systems and anticipate changes one needs to understand (_ask system owners_)
- How are the source systems set up
- What kind of changes are to expect


# Ingestion from Source Systems

This is commonly the first step -- moving data from source systems to the place where data can be manipulated and managed

_Qestions_ to address when deciding on data ingestion infrastructure
- __Batch__: Frequency of data ingestion (e.g., batch ingestion every so often)
- __Streaming__: Igest data in a constant stream of events in near-real-time. Data is available shortly after it is prduced.
    - Tools: _Event-streaming platform_ or _message queus_
    - _Cost/benefit analysis_ is adviced before implementing due to increased complexity
    - _Example usecase_: real-time anomaly detection

__Trade-offs__. 
- Is there a benifit from straming data (e.g., ML models are usually trained using batch data)
- What is the increase in cost/maintenance/downtime 
- Commonly two approaches co-exist within one project


# Storage

__Raw Hardware Ingredients__: _Bottom layer_
- Magnetic Disks (backbone of the system (cheaper))
- Flopy Disks
- RAM (faster read and write, but not persistent -- volitale, expensive)
- __CLOUD Storage__ has the follwoing added ingredients:
    - Networking
    - Serialization
    - Compression
    - Caching

__Storage Systems__ (built on top of physical ingredients) _Second Layer on top of the basic ingreients_
- Database Managment Systems
- Object Storage
- Apache Iceberg
- Cache/Memory-based storage
- Straming storage

__Storage Abstructions__: _Top layer of the hierarchy_
- Data Warehouse
- Data Lake
- Data Lakehouse

__Configuration parameters__:
- Latency 
- Scalability
- Cost


# Data Transformations: Queries, Modelling, Transformation

Example:  
A _downstram user_, analyst, needs to report on daily sales of some products. He needs thus data such as: customer IDs, product names, prices, quantities, time of sales, etc. Analysts needs the data be accessible via a simple SQL queury. Dat aneeds to be tranform and pre-processed. 

__Query__: read data from database or other sotrage systems e.g., data warehouse. Language: usually, SQL. Preprocessing is done using
- _Data Clearning_: (DROP, TRUNCATE, TRIM, SELECT DISTINCT)
- _Data Joining_: (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, UNION)
- _Data Aggregation_: (SUM, AVG, COUNT, MAX, MIN, GROUP BY)
- _Data Filtering_: (WHERE, AND, OR, IS NULL, IS NOT NULL, IN, LIKE)

There are many ways to write quieries. Must be efficient and correct or it will break the storage systems. 

__Data Modelling__: chosing coherent _structure_ for the data. For example data is often stored as _Noramilzed data_ in relational database as a separate complexly interconnected databases. Commonly you need to _De-normalize_ data so that downstream user can use it more easily


__Data Transformation__: manipulations that enhance the data and save it for downstream use  
Examples:
- Adding timestamps to record data
- Enriching records with additional fields and calculations
- _de-normilize data_ and apply _large scale aggregations_, add features for ML models

Commonly, multiple transformations occure during the data processing from source to the downstream user


# Serving Data

This is a final stage of the data engineering life-cycle.  
This is where the data has value for use cases.  

__Analytics__: 




