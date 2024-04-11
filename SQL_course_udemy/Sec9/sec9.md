# Logical expressions and functionality within SQL

- CASE
- COALESSCE
- NULLIF
- CAST
- Views
- IMPORT and Export Functional

## CASE 

`!` Execute case if conditions are met

__General Syntax__:
```sql
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE some_other_result
```

Return many statmenets based on the ocndition

Example:
```sql
SELECT a 
CASE WHEN a = 1 THEN 'one'
     WHEN a = 2 THEN 'two'
ELSE 'other' AS label
END
FROM test;
```
for relabling a column name 

__CASE EXPRESSION__ 
1. Evaluate expression
2. Compare the result with the value 
__GENERAL SYNTAX__:
```sql
CASE expression
    WHEN val1 THEN res1
    WHEN val2 THEN res2
    ELSE res3
END
```

__NOTE__: general case sytax is more flexible than _case expression_. 
It also allows for several conditions after WHEN 

__EXAMPLE__:

```sql
SELECT customer_id,
CASE
    WHEN (customer_id <= 100) THEN 'Premium'
    WHEN (customer_id BETWEEN 100 and 200) THEN 'Plus'
    ELSE 'Normal'
END 
AS cusomer_class
FROM customer
```

Then, operations can be performed on the result

## COALESCE

Expects all arguments in order to be evaluated in and returns the first _non NULL_ value. 

Commonly used to repalce NULL with zero for calculations __without__ altering the table itself

## CAST

Converts from one datatype to another

__General syntax__:
```sql
SELECT CAST (date as TIMESTAMP) 
FROM table
```
Commonly applied to an entire column.  

__EXAMPLE__:
```sql
SELECT CHAR_LENGTH(inventory_id AS VARCHAR) FROM rental
```
Note, here we first need to conver to a char before getting length

## NULLIF

Takes two imputs; return NULL if they are equal and the first arg, if not equal

Usefull when NULL cases an error