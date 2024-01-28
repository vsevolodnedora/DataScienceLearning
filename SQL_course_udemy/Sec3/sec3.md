# Intro to GROUP BY

- aggregate data and consdier data distributed by category

### Aggregate Function

- Main idea: take _multiple_ inputs and return a _single_ output

__Examples__:
- `AVG()` - average (returns a float); use `ROUND()` to specify precision
- `COUNT()` - number for values
- `MAX()` - return max
- `MIN()` - return min
- `SUM()`

`!` They work __only__ in he `SELECT` or `HAVING` clause; 

__Example__:
```sql
SELECT MIN(replacement_cost), MIN(replacement_cost) FROM film;
SELECT ROUND(AVG(replacement_cost),2) FROM film;
```

### Use GROUP BY

- Aggregate columns by a category

For a GROUP BY a _categorical_ columns must be chosen.  
Categorical columns are _non-continous_, but __Can be__ numerical, even floats.  

- Idea is to split larger tables into sub-tables and perform some form of _aggregate functions_. 

- Basic Syntax:  
```sql 
SELECT category_col, AVG(data_col) FROM my_table 
GROUP BY category_col
```

`!` GROUP BY __must__ appear right _after_ the FROM or WHERE statement

__Example__:
```sql 
SELECT category_col, AVG(data_col) FROM my_table 
WHERE category_col != 'A' 
GROUP BY category_col
```

`!` In the SELECT statement, columns must _either_ have an _aggregate function_ or be in the GROUP BY call

__NOTE__: We selected the column 'category_call' above, so we have to include it into the GROUP BY statement. The __exception__ is when we include the _aggregate function_ as in this example

__Example__:
- Total sum of sales __per__ devision, and company
```sql 
SELECT company,division SUM(sales) FROM finance_table 
GROUP BY company,division
```
where again we used two columns _without_ aggregation, by we included _both_ into the GROUP BY

__NOTE__: do not use the column with aggregation in WHERE statement (use HAVING instead)

__Example__:
```sql 
SELECT company,divisin, SUM(sales) FROM finance_table 
WHERE division IN ('marketing','transport')
GROUP BY company,division
```

- _sorting by an aggregate_ is done as follws:
```sql 
SELECT company, SUM(sales) FROM finance_table 
GROUP BY company
ORDER BY SUM(sales)
LIMIT 5
```

- The simplest GROUP by is _equivalent_ to SELECT DISTINCT() as 
```sql
SELECT customer_id FROM payment
GROUP BY customer_id
```

- Example where we group by by the total 
```sql 
SELECT customer_id, SUM(amount) FROM payment
GROUP BY customer_id
ORDER BY SUM(AMOUNT) DESC
```

- Total Amount spend _per_ customer, _per_ staff
```sql
SELECT customer_id,staff_id, SUM(amount) FROM payment
GROUP BY customer_id,staff_id
ORDER BY SUM(AMOUNT) DESC
```

#### GROUP BY using 'date' 

__NOTE__: data contains the _timestemp_ with day and time.  
There you use a special time as follw to get the _date_ only
```sql
SELECT DATE(payment_date) FROM payment;
```
