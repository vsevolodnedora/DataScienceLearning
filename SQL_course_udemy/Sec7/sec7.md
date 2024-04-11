# Assessment Test

## Restoring the database

__NOTE__ this database has two chemas: 
- Public schema
- cd. schema

When quiring data from a specific schema, use 
```sql
SELECT * FROM cd.bookings 
SELECT * fro
```

Using new database with 
- facilities 
- bookings 
- members

```sql
select * from cd.facilities
where name ilike '%Tennis%' -- like - anywhere in athe name, % for any character number

select * from cd.facilities
where facid in (1,5) -- Note list in SQL is ()

select * from cd.members
where joindate >= '2012-09-01'

select max(joindate) from cd.members

-- Total slots booked per facilities
select cd.bookings.facid, sum(cd.bookings.slots) as total 
from cd.bookings
where starttime >= '2012-09-01' and 
starttime <= '2012-10-01' -- be carefull with between operatior
group by facid
order by sum(cd.bookings.slots)

-- different task
select cd.bookings.facid, sum(cd.bookings.slots) as total 
from cd.bookings
group by cd.bookings.facid
having sum(slots) > 1000 -- use having for aggeate func
order by cd.bookings.facid
```

- Compleq qeury 
```sql
select cd.bookings.starttime, cd.facilities.name 
from cd.facilities
inner join cd.bookings 
on cd.bookings.facid = cd.facilities.facid
-- filter on start times on september and facility id
where cd.facilities.facid in (0,1) -- select tee
and cd.bookings.starttime >= '2012-09-21'
and cd.bookings.starttime < '2012-09-22'
order by cd.bookings.starttime
```