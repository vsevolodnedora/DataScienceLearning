# Advanced SQL Commands
- timestamps, EXTRACT
- Math Funcitons
- String Functions
- Sub-quiery
- Self-Join

### Timestamps and Extract

Time information keys:
- TIME - contains only time
- DATE - contains only date
- TIMESTAMP - contains data and time
- TIMESTAMPTZ - contains dat, time, timezone

__NOTE__: depending on what table is used for, some information can be either needed or not. 

#### Functions realted to specific data types 
- TIMEZONE
- NOW
- TIMEOFDAY
- CURRENT_TIME
- CURRENT_DATE

__NOTE__: to see a banch of runtime parameters, use SHOW ALL instead of 

- SHOW TIMEZONE returns the current time of the computer
- SELECT NOW() give me the timestamp information for now (GMT time)
- SELECT TIMEOFDAY() returns the current time,day,date in a string format
- SELECT CURRENT_TIME returns just time
- SELECT CURRENT_DATE returns just date

This is important when new data is added to the table

### Extract information from time-based data types

- EXTRACT() allows to extract a sub-combponent of a date value (YEAR, MONTH, DAY, WEEK, QUARTER)
    - EXTRACT(YEAR FROM date_col)
- AGE() calcualtes the current age given a timestamp, returns a difference between the NOW() and the timestamp 
    - AGE(date_col) 
- TO_CHAR() General function to convert date types to text
    - TO_CHAR(date_col,'mm-dd-yyyy')

### Examples:

- Get years for given timesteps
```sql
SELECT EXTRACT(YEAR FROM payment_date)
FROM payment
```

- Get how _old_ a given timestemp is:
```sql
SELECT AGE(payment_date)
FROM payment
```
