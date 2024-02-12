# Lec 1 Overview of tutorials

Challenges are based on a DVD store database

Challenges are based on 
- repeating the lecture material (without solution)
- new challenge, with solution in the next lecture

See _ScreenShot_ for overview on SQL commands


_Challenge Structure_:
- Business Situation
- Challenge Question
- Expected Answer
- Hints
- Solution


# SQL Statement Fundamentals

SQL syntax can be applied to any SQL database, not only to PostgreSQL

Database is composed of tables. 


### SELECT 

- retrieve data from the table in the database
- usually sequal statmenets are 'Capitlized'
- Example: `SELECT colname FROM tablename`

__NOTE__: SQL starts by chosing the database. This is already set as editro is openned for this database. Then it looks for a _table_ that is required, than for a _column_ that is required. 

- Selecting multiple columns: 
Example: `SELECT col1, col2 FROM tablename`

- Selecta __all__ columns from a table:
Example: `SELECT * FROM tablename`  
__NOTE__: in general it is not used, as it requires a lot of traffic to query data. 

#### Challenge 1: 

- Send a promotional email to existing customers: We need names, emails. 
- Use `SELECT` to grab first, last name and email adress. 
- Expected answer / solution
- `SELECT first_name,last_name,email FROM customer;`

### SELECT DISTINCT

`!` This is the samme as "select unique values in a column". 

- It is simiat to _remove duplicates_ statement

- __Example__: `SELECT DISTINCT col FROM table`
- __Example__: `SELECT DISTINCT(col) FROM table`
- __Example__: `SELECT DISTINCT(release_year) FROM film;`

### COUNT

- `COUNT` function returns number of rows _that match specific condition_ of a query
- `COUNT(col)` is applied to a specific column, while `COUNT(*)` is applied to all columns

__NOTE__: `COUNT` requires parenthesis ()

__NOTE__: For a given table `SELECT COUNT(col1)` = `SELECT COUNT(col2)` = `SELECT COUNT(*)`, as number of _rows_ is the same in the column. 

- It is usefull in combination with other commands, like `DISTINCT`

- __Example__: `SELECT COUNT(DISTINCT(col)) FROM table` e.g., count unique results 

### SELECT WHERE

- Specity conditions on columns for rows to be returned

Basic syntax is:
`SELECT col1,col2 FROM table WHERE conditions`

__NOTE__: `WHERE` comes immedeately after `FROM` _of_ the `SELECT`

Conditions are used to _filter_ the rows. 

- __Comparison operators__: compare column value to a limit. 
    - Operators are basic math: '=,<,>,<=,...'
    - Logical operators are 'AND' 'OR' and 'NOT'

- __Example__: `SELECT name,choice FROM table WHERE name = 'David'`

#### Challenge:

__Task__
- Find cusntomer _email_ for name _Nancy Thomas_.  


### ORDER BY

Resolut of a quiry can be machine dependt. 

We can _sort_ column by the value (alphabetic or nuemric)

Syntax
- `SELECT column_1,column_2 FROM table ORDER BY column_1 ASC/DESC` (ASC is default)

`!` This operatiom should be at the _end of the quiery_, SQL performs it __last__.  

`!` We can mave multiple ORDER BY rows for output with multiple columns

__EXAMPLE__:
`SELECT store_id,first_name,last_name FROM customer ORDER BY store_id,first_name ASC`

__NOTE__: statements are chaned, while ORDER BY is used only _once_!

`!` Each colmun may have its own order as `ORDER BY store_id DESC, first_name ASC;`

__NOTE__: we can ORDER BY columns that _we do not SELECT_ !


### LIMIT

Limits the number of rows returned. Usefull to limit the view of the table. This is the __last__ command the SQL will execute

__Example__:

`SELECT * FROM payment ORDER BY payment_date DESC LIMIT 10; `

That can answer a business-like question 'give me the top ten most recent payments...'

- Top N is `ORDER BY ... DESC LIMIT N;`
- Bottom N `ORDER BY ... ASC LIMIT N;`

Also `LIMIT 1` allows to quickly view the table, its contnet


#### Challange Order by

Business situation
- Reward first 1 paying custromers
- Find Customers ID who made a payment

- Result
`SELECT * FROM payment WHERE amount != 0.0 ORDER BY payment_date ASC LIMIT 10;`

`!` Note, there are column names that are same as sequal commands. One has to be carefull not to use these words


### BETWEEN

- Match value between a range.  Similar to WHERE, but with both limits are inclusive e.g., => <=. 
- NOT BETWEEN is equivalent to both < >, __NOTE__ they are not inclusice 

__NOTE__: BETWEEN can be used with dates; but dates must be formated with __ISO 8601__ standard YYYY-MM-DD 

__Example__: 
- `SELECT * FROM payment WHERE amount BETWEEN 8 AND 9;`

### IN 

Check for multiple possible options (sequience of ==), it is like python 'in'

- NOT ()
- NOT IN ()

- __Example__: ` SELECT COUNT(*) FROM payment WHERE amount NOT IN(0.99,1.98,1.99);`


### LIKE and ILIKE

`!` Pattern matching with string data. 

Where we want to match agains general patterns in a string: e.g., all emails which end with 'gmail.com' or all names that being with A. 

The pattern matching is done using so-called _wildcard characters_,
- `%` matches any sequence of characters 
- `_` matches any single character

__NOTE__ LIKE is case sensitive while __ILIKE__ is case insensitive 

Compled patterns can be created using multiple of these as `LIKE '_her%`

__NOTE__: Postrgre SQL supports all __regex__ as in python

__Example__: `SELECT * FROM customer WHERE first_name LIKE 'J%' AND last_name LIKE 'S%'`
SELECT * FROM customer WHERE first_name NOT LIKE '%er%' AND last_name NOT LIKE 'B%' ORDER BY last_name 


### General Challenge

