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