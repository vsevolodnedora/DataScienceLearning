# Intro to Data Engineering

Given by co-author of the book "Fundamentals of Data Engineering"

Data engineering is needed to do data science

Data engineer is a supporting role. 

Data-centri AI -- discipline to systematiclly engineer data used to built AI systems. 

Things to learn:
- Framework
- Principles
- AWS for practicle exercises (cloud)


# Overview

Imagine a job with no data infrastucture  

Designing and building data systems

Data achitecture is not different between applications

Workflow:
> Stakeholder Needs -> System Requirements (functional and non-functional) -> Technology choices (injection, storage, transformation)

Understand the requirements first before implementation. 

# Data Engineering Defined

> Data Engineering is a development, implementation and maintanence of systems and processes that take in raw data and produce high quality, consistent information that supports various downstream use casees.

Data engineering lies at the intersection _undercurrents_ of data:
- D. security, 
- D. managment, 
- DataOps, 
- D. arcitecture, 
- D. orechastration and 
- software engineering. 

__Data Engineering Lifecycle__:
Generation -> Storage [Injestion, Trnasformation, Serving] -> Use Cases [Analytics, ML, Revers ETL]

> Data Pipeline is the combinatioon of architecture, systems and processes that move data through the stages of the data engineering lifecycme

# Data Engineer Among Other Stakeholders

It is important to understand the __downstream task__ first. These include:
- Analysts (how often?, What info and what pre-processing/conversion?, Tolerable latency?)
- Data Scientists
- ML engineers
- Selespeople
- Marketing professionals
- Executives

__Stakeholders__:
- Upstream, eg., data generation, application software engineers (hat volume, hat frequency, format, compliences)
- Downstream (analysis, business)

> Business Value: go where the money is and look for business value, not for interesting technology:Revene growth, percieved value -- by stakeholders

Once the system, goal, is identified, the _system requirements_ need to be set

# System Requirements

- Business Requiemnts (high level goals of business)
- Stakeholder requirements (needs of individuals within the company)

System requirements:
- __Functional__: 'What system needs to do?'
  - Provide regular updates to databases
  - Alert users about anomalies
- __Non-functional__: 'How the system acomplishes what it needs'
  - Technical specifications of injection and orchestration
  - How to meed end users' needs
  
__Requirements Gathering__ from _downstream_ stakeholders (or their business goals)
- Business and stakeholders requirments
- Features and Attributes
- Memory and Storage Capacity

> Key Performance Indicator (KPI) is a measure of a company's progress towards an objetive
> Profit and Loss (P&L) statement: a financial statement that summarizes the revenues, costs, and expenses over a specific period of time
> Enterprise Resource Planning (ERP) Migration: moving an organization's resources and proceses from one software system to another

# Requirements Gathering (Conversation with a data scientist)

What is day-to-day tech stack? -- [Snowflake](https://www.snowflake.com/en/emea/), -- __data warehouse-as-a-service solution__ and used for
- Data Storage and Preparation (repository for raw data from various sources)
- Data wrangling using SQL queries (or python)
- EDA via SQL
- Data pipelinining via ETL tools (Fivetran, Matillion, dbt)
- Feature engineering (via SQL)
- ML via __Snowpark__ (their developer network) that suppots python
- Deployment of ML via __DataRobot__, __H2O.ai__

Example of Snowflake useage:
- __Injest data__ from raw files, dataases
- __Transform data__ vis SQL queries for imputation, aggregation
- __Feature engineering__ and building ML models dorectly in Snowflake
- __Model Development__ via Snowpark or external libraries (python) 
- __Model Deployment__

Example task for DS:
- Real time analysis of market sales by region (what is bought and where)
- Data access: data dump (needed and not needed) once a day and manuall pull required
- Extensive data preprocessing: gather, clean, impute, save in usefull formats
- Pain points: data is messy and preprocessing takes a long time (missing entries, wrong values. Foremats cahnge and procesion breaks. Data must be usefull for dashboards which requries a lot of wrangling. Stakeholders require data in real time while because of dayly updates and wrangling issues, there is a large delay

Summarize the issues of a DS. Ask what DS's stakeholders want? (understand the needs of the end customer), e.g., effectiveness of market campains, trends in sales, recommendation engines, etc. 

__Identify usecases for data__:  
- Dashboards for data (product sales by category and region) for trand analysis
- Recommender machine to pass the most popular items to software team so that they push more adds for this product

__Practical Conclusions__:  
- More frequent and more targeted data injestions (dumps)
- Automate and ochestrate data serving (in the right formats) for the dashboard and recommender

# Translate Stakeholder Needs into Specific Requirements

__Key Elements of Requirements Gathering__
- Learn what existing data systems orsolutions are in place
- Learn what _pain points_ or problems there are with existing solutions
- Learn what actions stakeholers plan to take with the data (their goals). This helps identify non-functional requirements. Going down the cahin and finding out what stakeholders want to learn from data may help defining out non-functional requirements like data quality, frequency, etc.
  - Repean what you understood back at stakeholders
- Identify additional stakeholders that need to be contacted for additional information

Note, it is common that production database is not available for analytics as to not to compromise it. 
Note, database schema may change. 

Additiona converstaion might include talking to database operating software engineers to ask about how the data injestion solution might look like. What disruptions are expected and hwo to have advance noticies about schema changes. 

__Functional Requirement__:
- Injest, transform & serve data in the format that is required

__Non-Functional Requirement__:
- latency (note, that real-time may have different values for different tasks) 

# Thinking like Data Engineer

After requirements gathering, -- decisions on how to build the system to meet the requiremens 

1. Identify business goals & stakeholder needs
   - Find staeholders that need your solutions
   - Explore exisiting systems and stakeholder desires to improve them
   - Ask stakeholder about their goals and tasks have for or do with the data product
2. Define system requirements
   - Translate stakeholder needs into _functional_ requirements (what systems must be able to do to meet the requriement)
   - Define _non-functional_ requirements (how the system should do what it has to do)
   - Document and confirm requirements with stakeholders 
3. Chose tools and technologies to build the system
   - Identify tools and tech that meeds the _non-functional_ requirements
   - Perform cost/benefit analysis and chose between comparable tools and tech
   - Prototype and test your system, align with stakeholder needs. Iterate on the prototype. 
4. Build, evaluate, iterate and evolve
   - Build and deploy the data system into production
   - Monitor, evaluate and iterate on the system to imporve it
   - Evolve your system based on stakeholder needs



# Data Engineering on the Cloud


   
