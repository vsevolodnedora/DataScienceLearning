# Creating databases and tables

## Data Types
__Categories__:
- Boolean; True or False
- Char (varchar...)
- Numeric 
- temporal (date,time)

__Other data types__:
- UUID - Universally Unique Identifiers (algorithmically unique code for a unique row)
- Array - array of strings, numbers, etc
- Json
- Hstor (key-value pair)
- geometric data

`!` creating a data table it is important to consult the documentation so that limits of the data type are respected. 

__Example__: storying a phone number: 
- Store as numeric - go to numeric options - check the valid size e.g., _smallint_, _integer_ or _bigint_... Note, however, a phone number does not require arithmitic, so we can store it as strings
- Search for _best practicies online_ for storing a certain data type

`!` Plan for long-term storage. Information cannot be added in the past, so it is better to allocate more and add more in case it is needed in the future.  


## primary Key and a Foregn Key

A _primary key_ is a column / group used to identify a row _uniquely_ in a table (it must be non-NULL). Integer and unique. 
IT allows to join tables together.  
Recall we had "customer_id" column that had a marker `[PK]` which stands for Primary Key.  

A _foregn key_ a field or a group of fields that _uniquely identifies_ a row in _another table_.  
A foreign key is defined in a table that references to the _primary key_ of another table.  

A table is called _a referncing table_ or a _child table_ if it contains the foregn key.  
A table to which a foreign key refers to is called _referenced table_ or a _parent table_.  

A table may have multiple foreign keys.  
Recall we had a table with payment_id as primary and customer_id as a foreign key. Foregn keys may be _non unique_ in the referencing table.

These keys are natural choices for joining tables.  

`!` when creating tables and columns we can use constraints to define columns as _primary_ or _foreign_ keys or attaching a _foreign key relationship to another table.  

`!` a foreign key is a _constraint_ on the column

See 'constraints' in /database/schemas/tables/my_table/constaints/, there there are primary and foreign keys. 
Select a key and go to _dependencies_ tab to see what tables it is referencing


## Constraints 

`!` Constraints are the rules enfored on data columns on tables.  
- Constraints are used to prevent _invalid_ data from entering the database  

Constaraints are devided into 
- Constraints on a Column
- Constraints on a Table

__Examples__:
- NOT NULL - does not allow Nulls to be enteres
- UNIQUE - ensures that all values are unique (usefull for e.g., IDs)
- PRIMARY Key (set up as a constraint) uniquly identifies each row/record in a database
- FOREIGN Key (constraints data based on columns on other tables)
- CHECK Constraintd - Ensure that all values in a column satisfy certain conditions
- EXCLUSION constraint - ensures that if any two rows are compared on the specified column or expression using the specified operator, not all of hese comparisons will return TRUE
- CHECK(condition)
- REFERENCES - to constrain the values

__Table Constraints__:
- UNQIEU(column_list) - for multiple columns (table constraints)
- PRIMARY KEY(column_list) (also for multiple columns)

## Creating a table 

```sql 
CREATE TABLE table_name ( -- chose table name
    column_name TYPE column_constraint, -- set column name and its data type and constraints (use comma that separates the columns)
    column_name TYPE column_constraint, 
    table_constraint table constraint -- table-level constraint
) INHERITS existing_table_name; -- add relationship with other tables
```

- __SERIAL__ is a special datatype
    - In postgreSQL a _sequence_ is a special kind of database object that gnerates a sequence of integers
    - A _sequnce_ is ofter used as a __primary key column__ in a table
    - It logs an ID automatically when new data is added to the table
    - If row s removed later, _serial will not adjust_ and will keep its value
    - has several datarypes
        - smallserial 
        - serial
        - bigserial

__Example__: simple table
```sql
CREATE TABLE players(
    player_id SERIAL PRIMARY KEY 
    age SMALLINT NOT NULL
)
```

Craeting tables usin _queuey tool_
```SQL 
CREATE TABLE account(
	user_id SERIAL PRIMARY KEY, -- set utomatic counter
	username VARCHAR(50) UNIQUE NOT NULL, -- LIMIT FOR STR; UNIQUE
	password VARCHAR(50) NOT NULL,
	email VARCHAR(250) UNIQUE NOT NULL,
	created_on TIMESTAMP NOT NULL, -- WHEN ROW IS CREATED
	last_login TIMESTAMP -- NO CONSTRAINT
)
-- CREAT ANOTHER TABLE FOR A JOB
CREATE TABLE job(
	job_id SERIAL PRIMARY KEY,
	job_name VARCHAR (200) UNIQUE NOT NULL
)
-- LINK TWO TABLES VIA A PRIMARY KEY
CREATE TABLE account_job(
	-- create a foreign key as int and reference another table column to it
	user_id INTEGER REFERENCES account(user_id) -- NOT A PRIM. KEY; 
	job_id INTEGER REFERENCES job(job_id)
	hire_date TIMESTAMP
)
```

here we created 3 tables and linked them via the third tabel where we set up two foreign keys.  

### Inserting information :: INSERT

__General Syntax__:
- Inserting value into the table:
```sql
INSERT INTO table(column1,column2,...)
VALUES 
    (VALUE1, VALUE2, ...),
    (VALUE1, VALUE2, ...),
    ...
```

- Inserting from another table 
```sql 
INSERT INTO table(column1,column2,...)
SELECT column1,column2,...
FROM another_table
WHERE condition
```
__NOTE__: the columns must match _exactly_ -- __size and constrants__. 

__EXAMPLE__: adding one row to the table
```sql
INSERT INTO account(username,password,email,created_on)
VALUES 
('Jose','password','jose@mail.com',CURRENT_TIMESTAMP)
```

Making a connection using a foreign key
```sql
insert into account_job(user_id, job_id, hire_date)
values
(1,1,current_timestamp)
```
__NOTE__: we use ID for foreign keys


### UPDATE 

- Changing the values of columns in a table

_General syntax_:
```sql
UPDATE table 
SET column1 = value1,
    column2 = value2, ...
WHERE 
    condition;
```

__Example__: update last login with current timestamp
```sql
UPDATE account
SET last_login = CURRENT_TIMESTAMP
WHERE last_login IS NULL
```

- or using another column to update everything

```sql
UPDATE account
SET last_login = created_on
```

- Use another table to update new column values. Sometimes called 
_update joint_

```sql
UPDATE tableA
SET original_col = TableB.new_col
FROM tableB
WHERE tableA.id = tableB.id
```

- Return affected raws
```sql
UPDATE  ...
SET ...
RETURNING account_id, last_login
```

__EXAMPLE__: adding last login using value from another table 
```sql
update account_job
set hire_date = account.created_on
from account
where account_job.user_id = account.user_id
returning account.created_on,hire_date
```

### DELETE

Remove data from tableA if it matches with tableB (delete join)
```sql
DELETE FROM tableA 
USING tableB
WHERE tableA.id = tableB.id
```

- Delete __ALL__ rows from a table
```sql
DELETE FROM table
```

__EXAMPLE__:
Delete a row and return what row was deleted
```sql
delete from job 
where job_name = 'cowboy'
returning job_id, job_name
```

### ALTER TABLE

`!` Change existing table structure
- Adding, dropping or renaming columns
- Changing a column's data type
- set DEFAULT values for a column
- Add CHECK constraints
- Rename table

__General syntax__:
```sql
ALTER TABLE table_name
ADD COLUMN new_col TYPE
-- or 
ALTER TABLE table_name
DROP COLUMN new_col TYPE
-- or
ALTER TABLE table_name
ALTER COLUMN col_name
    SET DEFAULT value  -- option 1
    SET NOT NULL -- option 2
    DROP NOT NULL -- option 3
    SET constraint_name -- CONSTRAINT option 4
```

### DROP 

`!` Remove the column from a table 
__NOTE__ postgreSQL also removes all the indeces and constraints related ot the column  
__NOTE__ the additiona; _views_, _triggers_ or _stored procedures_ are __NOT__ removed without additional CASCADE clause.  

__General syntax__:
```sql
ALTER TABLE table_name
DROP COLUMN if exists col_name CASCADE, -- REMOVE COL with dependencies
drom column col_two
```

### CHECK

`!` allows to constrain customized constraints for columns  

__General syntax__:
```sql
CREATE TABLE example(
    ex_id SERIAL PRIMARY KEY,
    age SMALLINT CHECK (age > 21)
    parent_age SMALLINT CHECK (parent_age > age)
);
```

__Example__:
```sql
create table emplyees(
	emp_id serial primary key,
	first_name varchar(50) not null,
	last_name varchar(100) not null,
	birthdate date check (birthdate > '1900-01-01'),
	hire_date date check (hire_date > birthdate),
	salary integer check(salary > 0)	
)
```




