# Intro to JOINs

### AS statement

- AS is just an alias for names

__EXAMPLE__:
- This will return a result where the output called "rental_prices"
```sql 
SELECT amount AS rental_price FROM payment; 
```


- Return a aggregate renamed as: 
```sql 
SELECT SUM(amount) AS total_price FROM payment; 
```

__NOTE__ the "AS" operator is executed at the _very end_ of the query.  

`!` Do not use AS operator _inside_ WHERE operator. Use it only in SELECT part

__EXAMPLE__: 
We cannot use HAVING or WHERE with this alias, so use 
```sql
SELECT customer_id, SUM(amount) 
	AS total_amount 
FROM payment 
GROUP BY customer_id
HAVING SUM(amount) > 100
```


### INNER JOIN

Joins allow to combine multiple tables together. The main difficulty is to see how to join tables that do not have same keys. 

`!` Inner join will return the result where the key matches in _both tables_. 
Set of records in both tables.  
Imagine two circles intercecting and only _mutual region_ is shaded. 
`!` INNER JOIN is __symmetrical__

```sql
SELECT * FROM TableA 
INNER JOIN TableB 
ON Table1.col_match = TableB.col_match 
```

To avoid duplicating data, select what _table_ to take the duplicate name from: 'Logins.name'
```sql
SELECT reg_id, Logins.name, log_id 
FROM Registration
INNER JOIN Logins
ON Registrations.name = Logins.name
```

__NOTE__ if a column existrs only in one table, PostgreSQL will find which one, so no need to attach the table name. Otherwise, if column exists in both, one has to specify which table to take the column from. 


### Full OUTER JOIN

Here we account for data present in only one table.  

`!` FULL OUTER JOIN grabs everything from both tables, so in the wienn diagram this is the area of both curcles combined. 

`!` It is also symmetrical

```sql
SELECT * FROM tableA 
FULL OUTER JOIN tableB
ON tableA.col_match = tableB.col_match
```

__NOTE__ data that is missing is filled with NULL values

##### Select Data Unique to both tables:
In Wienn diagram it is the area of two cirles _excluding_ the itersecting region
```sql
SELECT * FROM table1 
FULL OUTER JOIN table2
ON table1.col_match = table2.col_match
WHERE table1.id IS NULL OR table2.id IS NULL -- specify values; Oppisite to Inner Join
```
- This is agan _symmerical_

__EXAMPLE__: Make a quiry that selects unique info from customer and payment, so that there are no customers that did not make payments. 
```sql
SELECT * FROM customer
FULL OUTER JOIN payment
ON customer.customer_id = payment.payment_id
WHERE customer.customer_id IS NULL 
	OR payment.payment_id IS NULL
```
__NOTE__: the result should be an empty list, but I do get results...

So such joins can be used to filter out data that should not be in the final result if it is not on one of the columns or not in both columns


### LEFT OUTER JOIN

