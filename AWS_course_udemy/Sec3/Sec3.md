# What is Cloud computing

### Traditional overivew

- Websides work as: Client -> Network -> Server, where `packets` are routed.
- IP adresses allow _server_ to find _client_ and vise vers. (This is similar to mail system)

A server is composed of a 
- CPU for calculation
- RAM (fast memory)
- Storage for data
- Dabase for structured data (for searching and querying)
- Network: routers, switch, DNS server...

- __Network__: an infrastructure that connects _routers_ and _servers_. 
- __Router__: a networking device, forwarding packets between a computer and a _network_. 
- __Switch__: Forwards a packet to a correct _server_/_client_ in the _network_. 

The structure is 

`!` PC client <-> Router <-> Switch <-> MANY PCS (network)

Data center is comprised of servers and is a natural evolution of a single machine system. 

Disadvantages of classical IT data cetner infrastucture is 
- Expensive to maintain (rent...)
- Energy expenses, maintaince
- Limited scaling
- Adding/replacing is time consuming
- Reauries team to maintain the server
- Possible physical damage.

__Cloud__ Is a possible solution. It is an _on-demand_ delivery of compute power, database storage, applications and IT resources.  
- Usually accompnaied by _pay-as-you-go_ pricing.  
- Allows to allocate the right type and size of compute resources. 
- Almost instantaneous access
- Has interface for easy access of servers, storage, databases, application services
- AWS owns and maintains network-connected servicies

`!` Essencially, a __cloud__ is a datacenter, owned and maintained by someone else, from 
resources and services can be fetched _on demand_
- Examples: gmail, dropbox, google drive, etc...

__Cloud types__:
- __Private Cloud__: used by a single organization, full control, but managed by someone else
- __Public Cloud__: e.g., Azure, Google Cloud, AWS; owned and operated by an _external provider_. 
- __Hybrid Cloud__: some servers are private some are public. Control over sensitivie data on a private infrustructure. 

__Five Characteristics of Cloud Computing__: 
- __On-demand self-service__
- __Broad network access__
- __Multi-tenancy and resource pooling__ (other customers of AWS can share infrustructre without violating security or proviacy, using the _same_ physical resources). 
- __Rapid elasticity and scalability__ allows quick scaling on demand
- __Measured service__ (pay only for resources used)

__Six Advantages of Clound Computing__
- __Trade capital expense__ (CAPEX) for operational expense (OPEX): Hardware is not payed for but 'rented'. Reduction of _Cost of Ownership_ (TCO) And _Operational Expense (OPEX)
- __Massive Exonomy of Scale__ prices are epxected fo go down, as the scale of provider grows
- __No need to guess capacity__: scale automaticall on measured usage
- __Increased speed and agility__
- __No need to maintain the entire datacenters__ 

__Problems soved by the Cloud__:
- __Flexibility__: resources are given when needed
- __Cost-Effectiveness__: pay as you go
- __Scalability__: easy increase in hardware teir, infustructure
- __Elasticity__
- __High-availibity and fault-tolerence__: use many datacenters
- __Agility__: rapid development, test, launch software


### Types of Cloud COmputing

- __Infrastructure as a Service__ (IaaS): provide building blocks for cloud IT, providde networking, computers, data storage space; have Highest level of flexibility; parallel with traiditional on-premises IT
- __Platform as a Service__ (PaaS): organization does not need to manage the infrastructure; it can focus on the deployment and managment of applications; 
- __Software as a Service__ (Saas).

_Comparison_: 

What one has to managed with difference approaches

| On Premis   |     Iaas    |    PaaS     |    SaaS     |
| ----------- | ----------- | ----------- | ----------- | 
| application | application | application |             |
|    data     |    data     |    data     |             |
|   runtime   |   runtime   |             |             |
|  middlware  |  middlware  |             |             |
|     O/S     |     O/S     |             |             |
| virt-ation  |             |             |             |
|   servers   |             |             |             |
|   storage   |             |             |             |
|  networking |             |             |             |

__Examples__:
- __IaaS__: Amazon EC2 (AWS); GCP, Azure, Rackspace, Digital Ocean, Linobe
- __PaaS__: Elastic Beanstalk (AWS); Heroku, Google App Engine (GCP), Widnws Azure
- __SaaS__: many AWS sercives (Rekognition for ML); Google Apps; Dropbox, Zoom


__AWS Pricing__:
- __Compute__: pay for _the exact compute time_
- __Storage__: pay for the _examct amount of data stored in the Cloud
- __Networking__: pay _only_ for data that _leaves the cloud_; __not__ _into the cloud


### History of AWS

- Launched internally in 02

- AWS allows to buid sophisticated, scalable applications
- Applicable to diverse set of industries
- Use cases: Enterpise IT, Backup & storage; Big Data analytics; Website hosting; Mobile & social apps
- Gaming servers

__AWS Global Infrastructure__:
- __AWS Regions__: has names like 'us-east-1'... it is a _cluster of datacenters_; most services are _region scoped_. 
    - _How to choose an AWS Region?_: It depends on 
        - __Compliance__ with data guverancne and legal reqiirements
        - __Proximumty__ (to reduce latencty)
        - __Awailable services__: check if region has the required servies
        - __Pricing__: different regions have different pricing
- __AWS Availablity Zones__: are stores within a region; _3-6_ zones in a region; each zone has several data centers. These availablity zones are _separated_ as to not to be affected by disasters. Zones are connected via _low latancy_ network. 
- AWS Dat Centers
- __AWS Edge Locations / Points of Presence__: AWS has _400+_ Points of Presence (10+ Regional Caches) in 90+ cities across 40+ countries; Content can be delived with low latency. 

__AWS Global Services__:
- __Identity and Access Managment__ (IAM)
- __Rout 53__ (DNS service)
- __CloudFront__ (Content Delivery Network)
- __Web Application Firewall__ (WAF)

__Region-scoped AWS Services__
- Amazon EC2 (IaaS)
- Elastic Beanstalk (PaaS)
- Lambda (Function as a Services, FaaS)
- Rekognition (SaaS)




