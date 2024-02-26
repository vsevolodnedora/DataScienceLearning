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

- Select data time to a text format (use specialized formats to organized the text)
```sql
SELECT TO_CHAT(payment_date, 'MONTH YYYY')
FROM payment
```

Or look up an exact format in the table

```sql
SELECT TO_CHAR(payment_date,'dd/mm/YYYY') 
FROM payment
```

See [formating](https://www.postgresql.org/docs/12/functions-formatting.html) for how to format a string

### Exerciese:

- Which are the monthes when payments occure

```sql
SELECT DISTINCT(TO_CHAR(payment_date, 'month'))
FROM payment
```
- how many payments occured on a monday

```sql
SELECT COUNT(TO_CHAR(payment_date, 'DY'))
FROM payment
WHERE TO_CHAR(payment_date, 'DY') = 'MON'
```

# Mathematical Functions

Check the pgsql documentation for exact list of functions 

### Examples
```sql
SELECT ROUND(rental_rate / replacement_cost,2)*100. 
FROM film 
```

# String functions and operators

See the documentation for the list of available functions

- String concatanation example:
```sql
SELECT first_name || ' ' || last_name AS full_name
FROM customer
```
- Creating an email : first letter forst name + last name + @comapny

```sql
SELECT lower(left(first_name,1)) || lower(last_name) || '@gmai.com' 
AS custom_email
FROM customer
```

# SubQuery

A sub query allows to construct more complex queries , performing a quicy on the result of another query

#### Example:
Get a list of sturctents that performed better than the average 
```sql
SELECT student,grade 
FROM grades
WHERE grade > ( -- subquery
    SELECT AVG(grade) 
    FROM test_scores
)
```

__NOTE__ subquery is always perform first. 
`!` An _WHERE IN_ operator can be used to check agains multiple results returnd 

### EXISTS
is used to check if a given row exists in a table 
```sql
SELECT column_name
FROM table_name
WHERE EXISTS
SELECT column_name FROM table_name
WHERE conditin
```

It is usefull to first write a subquery before the actual query. 

If a subqury returns 

#### Example:
- Get film titles return between certain dates 25.May.2005 and May..30.2005


# Self-Join 

Join a table to itself ( as if copy a table and joit it with itself )
`!` A table must be asliased

```sql
SELECT tableA.col, tableB.col
FROM table AS tableA
JOIN table AS table B ON
tableA.col = tableB.col
```

This is usefull for aliasing.  

### Example:
- Find all pairs of films that have the same length
```sql
select f1.title, f2.title, f1.length
FROM film as f1 
inner join film as f2 on
f1.film_id != f2.film_id
and 
f1.length = f2.length
```

