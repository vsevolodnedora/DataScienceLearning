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
> Analytics is the process of identifying key insights and patterns within data.

- __Business Intellegence__ - Exploring _historical data_ to extract business insights (in the form of reports and dashboards)
    - Campaign engagement
    - Regional Sales
    - Custormer experiece metrics
- __Operational Analytics__ - Monitoring _real time data_ for immediate actions
    - Real-time performance metrics for a webpage
    - Ingesting, transforming and serving event data
- __Embedded Analytics__ - External or customer-facing analytics
    - Analytics dashboards and data that is provided to a consumer (e.g., mobile app dashboards)
- __Machine Learning__ - Separate due to niche needs
    - Model training (feature stores)
    - Real-time Inference 
    - Track data history and lienage
- __Reverse ETL__ - use transformed data, analytics and ML output and feed it back to the source systems



# The Undercurrents of the Data Engineering Lifecycle -- Security

> Principle of lease privilige -- giving users or applications access to only the essential data and resources needed to do the job and only for a duration of this job

Examples:
- Avoid working from _root shell_ or using _super user_ priviliges if they are not needed  
- Ensiring _data sensitivity_ e.g., sensitive data, such as personal, medical, finantial data is anonymized and hidden unless required.  
- Avoid using sensitive data. 
- __Sequirity in the Cloud__
    - Identity and Acess Managment (IAM)
    - Encryption Methods
    - Networking Protocols
- Defensive mindset when access to sensitive data is requested. 
- Databases should not be exposed to general public. Sequirity best practicies should be used. 


# Data Managment

See Data Management Association International (DAMA) -- Book of Knowledge (DMBOK)

> __Data Managment__ - development and supervision of plans, programs and practicies that deliver, control, predict and enhance the value of data and information assets throughout thir life cycles. 

__Data Knowledge Areas for Data Managment__ (11 ares)
- __Data Guvernance__ (in the core)
- Data Modeling 
- Data Storage and operations
- Data integration and interoperability
- Data security
- Reference and master data 
- Metadata
- Data warehousing and Business
- Document & content Managment
- Data quality
- Data achitecture

See also _Data Governance: The definitive Guide_ 

> __Data Guvernance__, first and foremost, a data managment function to ensure the quality, integrity, security, and usability of the data collected by an organization

Consider Data Quality. 
- _High Quality Data_ - expected by stakeholders (schemas etc)
    - Accurate
    - Complete
    - Discoverable
    - Available in the timely manner
- _Low Quality Data_
    - Inaccurate
    - Incomplete
    - Missing Data


# Data Architecture

Road map or a blueprint for the data system. 

See Book: _Fundamentals of Data Engineering_

> __Data Architecture__ is the design of systems to support the evolving data needs of an enterprise, achieved by flexible and reversible decisions reached through a careful evaluation of trade-offs

Note:
- Evolving data needs. -- Data achitecture is an evolving effort
- Flexible and reversable deicsions -- architecture should be able to change or brought back
- Evaulation of trade-offs (scalability, performance, etc.)

Note, reversability benifits from cloud more than ever, as hardware on premises cannot be returned if not needed.

Principles of Good Data Architecture 
- Choose common components wisely (cross-team, cross-project compoennts)
- Plan for failure. 
- Architect for scalability (response for changing demand)
- Architecture is leadership
- Always be architecting
- Build lossely coupled systems (build from individual components)
- Make reversable decisions (if needs of organization changes)
- Prioritize security
- Embrace FinOps (finance, dataops etc -> cloud)


# DataOps

Improvs the development and quality of data products (similar to DevOps)  
It is a _set of cultural habits and practicies_. 
- Prioritizing communication and collaboration
- Continously learning from successes and failers 
- Rapid iteration

These techniques are borrowed from _Agile methodology_ (delivering work in iterative, incremental steps)

_Pillars of DataOps_
- __Automation__
    - DevOps (applied to software build) Continous Integration and Continous Delivery (CI/CD)
        - Automation of cycle: Build -> Test -> Integrate -> Deploy 
        - Faster review and deployment; less errors
    - DataOps (applies to data processing)
        - Automated change managment (code, configuration, environment)
        - Data procession and pipelining
        - _Pure Scheduling approach_ (low level automation for each task in the pipeline)
        - _Orchestration framework_ framework determins automatically which task depen on which and based on the frequncy launches them
        - Automatic verification
- Observability and Monitoring
    - Monitor systems
- Incident Response
    - Rapid identify root causes of failuers 
    - Identify technology and tools


# Orchestration

Initially a data pipeline can be set up as a manual project. For POC or simple project.  
Manual execution is common in prototyping.  
Once tasks are determning, _scheduling_ can be used.  
Pure scheduling however can cause cascade failuers if one task does not finish before the next one.  

Orchestration frameworks
- Apache airflow
    - Automate pipeline with complex dependencies
    - Monitor pipeline

Scheduling
- Time-based
- Event-based

Provides monitoring. 

Commonly, pipelines need to be set as __Directed Acyclic Graphs__ (DAG). 

Source systems -> Transfoormation step -> Storage -> Split -> Branch 1: transformation -> analytics Use Case; -> Branch 2: transformation for MLOps

DAGs are deployed within framework of choice.  


# Software Engineering

> __Software Engineeing__: design, development, deployment and maintenence of software applications

SQL, python, Bash  

Readable, translatablem production-ready code


---

# Practical Examples on AWS


---

# The Data Engineering Lifecycle on AWS

Source systems on AWS
- __Databases (most common)__
    - __Amazon Relational Database Service (RDS)__ 
        - Provisions database instances with the relational database engine of your choice
        - Simplifies the operational overhead involed with provisioning and hosting a relational database
    - __Amazon DynamoDB__
        - A serverless NoSQL database operational
        - Create stand-alone tables that are virtually unlimited in their total size
        - Has a flexible schema
        - Best suited for applications that require low-latency access to data
- __Streaming Sources__
    - __Amazon Kinesis Data Streams__
        - Set up as a source system streaming real-time user activities from a sales platform log
    - __Amazon Simple Queue Service__
        - Handle massages when building your own data pipelines outside of these courses
    - __Amazon Managed Streaming for Apache Kafka (MSK)__ 
        - Makes it easier to run Kafka workloads on AWS because the underlying infrastructure is managed for you

- __Injestion__
    - __AWS Database migration Service (DMS)__
        - Can migrate and replicate dta from a source to a target in an automatic ways
    - __AWS Glue__ 
        - Offers features that support data integration processes
    - __Amazon Kinesis Data Streams__ 
    - __Amazon Data Firehose__
    
- __Storage__
    - __Amazon Redshift__ (large standard storage)
    - __Amazon Simple Storage (S3)__
    - __Lakehouse Arrangments__ 
        - Combinations of various solutions that allow access structured data in the warehouse and unstractured data in an object data lake

- __Transformation__
    - __AWS Glue__
    - __Apache Spark__
    - __dbt__

- __Serving__
    - Analytics
        - __Amazon Athena__ (for quiring structured and unstractured data)
        - __Amazon Redshift__ (for quiring structured and unstractured data)
        - __Amazon QuickSight__ (dashboarding tools)
        - __Apache Superset__ (dashboarding tools)
        - __Metabase__  (dashboarding tools)
    - AI or Machine Learning
        - Serve batch data for training and vector databases

# Undercurrents on AWS 

> __Identity and Access Managment (IAM)__ setup roles and permissions for various services

- Sequirity

    - __IAM roles__
        - Give users/applications access to _temporary_ cridentials
        - Provide appropriate AWS API permissions to various tools or data storage Areas
    - Amazon Virtual Private Cloud (VPC)
    - Security Groups
    - Instance-level firewalls
- __Data Managment__ - Disconver, create and manage metadata for data stored in Amazon S3 or other storage and database systems
    - AWS Glue
    - AWS Glue Crawler
    - AWS Glue Data Catalog
    - __AWS Lake Formation__ - Centrally manage and scale fine-grained data access permissions

- __DataOps__
    - __Amazon cloudWatch__ - collects metrics and provides monitoring features for cloud resources, applications and on-premises resources
    - __Amazon Simple Notification Service (SNS)__ - sets up notifications between applications or via text/email that are triggered by events with the system
    - __Amazon CloudWatch Logs__ - store and analyze operational logs
    - __MONTECARLO__ (open source)
    - __BIGEYE__ (oepn source)

- __Orchestration__
    - __Apache Airflow__

- __Architecture__
    - __AWS Well-Architectured__ - Set of principles and practicies tailored for: Operational Efficiency, Sequirity, Performance Efficiency, Cost Optimization, Sustainability and Reliability. 

- __Software Engineering__
    - __AWS Cloud 9__ (IDE) hosted on Amazon Elastic Compute Cloud (EC2) i.e., it is an VM with an IDE installed on it
    - __Amazon Code Deploy__ that allows to automate code deployment
    - Version Control (GIT)