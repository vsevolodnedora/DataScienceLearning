# Data Architecture 

> Enterprise Architecture the designe of systems that support the change in the enterpaise

- __Business Architecture__
- __Application Architecture__
- __Technical Application__
- __Data Architecture__

Jef Bezon: One way and Two way doors and API mandate

Two-way decisions: changing storage type in AWS (can be reversed)  
Teams at Amazon are lossely coupled -> AWS 

# Conway's Law

An organization that desinges a system will have a structure that reflects the organization's structure

# Principles of Good Data architecture

1. Choose common components wisely 
2. Plan for failure
3. Architect for scalability
4. Achitecture is leadership
5. Always be acrhitecting 
6. Build loosly coupled systems
7. Make reversable decisions
8. Prioritize security 
9. Embrace FinOps

Shared sstorage layers allow multiple teams to use data
Identify components where teams can benefit from common components

Systems shoudl be built from _loosely coupled compoments_

# When you sysstem fail

- Failer mode: 
    - Availability (all systems are expected to fail and availability is )
    - Durability (ability of a storage system to withsdand data loss due to hardware failure, software loss or disasters)
    - Recovery Time Objective (max accaptable time for a service ot system outage)
    - Recovery Point objective (a definition of the accaptable state after recovery)

    Culture of security and principle of least privilage _Zero Trust Security_ 

    - Prioritize Security: 
        - Hardened-Perimeter Approach
    
    - Zero Trust
        - every access requires authentication (no walls or groups)

    AWS: EC2 or Spot Instances 

    Embrace FinOps:
        - Pay-as-you-go models:
            - Cost-per-query model
            - Cost-per-processing-capacity model

# Batch Architectures 

Tradeoffs in architecture

Most practical when real-time data is not needed and batch processing is sufficient

__ETL__: Data Sources -> Extract (batches) -> Staging area (transform) -> Load (where the data is needed)

__ELT__: Extract, Load, Transform

_Data Marts_: Additional transformations (sales, marketing, operations)

Choose Common Components

# Streaming Architectures

Data is injested in near-real-time fassion 
- APACHE kafka (event streaming platform)
- Apache Storm
- samza

- _Lambda Architecture_: Simultanous serving of:
    - Batch Processing -> Data Warehouse -> Serving Layer
    - Stream Processing -> NoSQL Database -> Serving Layer

Serving layer then aggregates the data via _mixed queries_ 

- _Kappa Architecture_: Using streaming platform for evverything (true event-based architecture)

# Architecting for Complience

- GDPR (EU in 2018) (personal identifiable inforamtion, -- concent from individuals and ability to remove it)
- Regulations of tomorrow
- HIPAA data 
- Financial data regulations
