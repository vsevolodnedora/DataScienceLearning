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


### Types of Cloud Computing

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


### Tour of the Console and Services in AWS

Login to your AWS account; 

"Upper right" - __Region Selector__ - chose region geographically close to your location 
    The closer the region the lower the latency

_loook_: at __Recently Visited Services__ (none at the beginning)

_Search_ allows to find services and, importantly, features, e.g., _rout 53_, as well as _blogs_, etc. 

If you go to __Route 53__ You see that it chages region to __Global__ -- this concole _does not require a region selection_. 

If you switch back to e.g., __EC2 service__ you will again be in your _local region_. 
__NOTE__: the view of the service _chages_ depending on the region.

Check [AWS global infrastucture](https://aws.amazon.com/about-aws/global-infrastructure/) that gives _additional information_ about regional services.

### Shared responsibility model in AWS
- A custromer is responsible for the securing __IN__ the clound (e.g., configuring the securing correlty). 
- AWS assures the securing __OF__ the cloud (e.g., infrustructre, software, internal infrustructure)

When using AWS, one agrees to 
- No illegal/harmfull or offensive use or content
- no security violations
- no network abuse
- no email or other message abuse

### QUEEZE

- Q3: Which Global Infrastructure identity is composed of one or more discrete data centers with redundant power, networking, and connectivity, and are used to deploy infrastructure?
- A3: This is the definition of Availability Zones.

- Q4: Which of the following is NOT one of the Five Characteristics of Cloud Computing?
- A4: Dedicated support agent to help you deploy applications (This is not one of the Five Characteristics of Cloud Computing. In the cloud, everything is self-service)

- Q5: Which are the 3 pricing fundamentals of the AWS Cloud?
- A5: Compute, Storage, and data transfer out of the AWS Cloud are the 3 pricing fundamentals of the AWS Cloud.

- Q6: Which of the following options is NOT a point of consideration when choosing an AWS Region? 
- A6: Capacity is unlimited in the cloud, you do not need to worry about it. The 4 points of considerations when choosing an AWS Region are: compliance with data governance and legal requirements, proximity to customers, available services and features within a Region, and pricing.

- Q7: Which of the following is NOT an advantage of Cloud Computing?
- A7: "Train your employees less" You must train your employees more so they can use the cloud effectively.

- Q8: AWS Regions are composed of?
- A8: AWS Regions consist of multiple, isolated, and physically separate Availability Zones within a geographic area.

- Q9: Which of the following services has a global scope?
- A9: IAM is a global service (encompasses all regions).

- Q10: Which of the following is the definition of Cloud Computing?
- A10: On-demand availability of computer system resources, especially data storage (cloud storage) and computing power, without direct active managment by the user

- Q11: What defines the distribution of responsibilities for security in the AWS Cloud?
- A11: The Shared Responsibility Model defines who is responsible for what in the AWS Cloud.

- Q12: A company would like to benefit from the advantages of the Public Cloud but would like to keep sensitive assets in its own infrastructure. Which deployment model should the company use?
- A12: Using a Hybrid Cloud deployment model allows you to benefit from the flexibility, scalability and on-demand storage access while keeping security and performance of your own infrastructure.

- Q13: What is NOT authorized to do on AWS according to the AWS Acceptable Use Policy?
- A13: You can run analytics on AWS, but you cannot run analytics on fraudulent content. Refer to the AWS Acceptable Use Policy to see what is not authorized to do on AWS.

