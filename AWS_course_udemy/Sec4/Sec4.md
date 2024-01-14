# Sec.4 IAM Introduction: Users, Groups, Policies

`!` IAM = Identity and Access Managment, __Global Service__
- This is where you _create users_ and _assign users to groups_
- It is used when _creating an account_ e.g., a _rout account_ which is created by default and _should not be used or shared_ 
    - It is only used to setup an account
    - It is also used to _create groups of users_ for an organization. IAM may contain _groups_. 
    - __Groups__ contain only users, not other grousp
    - A user _can_ belong to _multiple_ groups

`!` Users are created so that they can use AWS account safely. This is assured by giving them _permissions_.  
A User may have a `.json` documt attached to it, called __policies__ that is like

```python
{
    "version": ... 
    "Statement": 
    {
        "Effect": "Allow",
        "Action": "ec2:Describe",
        "Resources": "*"
    },
    ...
}
```

Where actions can be a _list_ with varios actions allowed for a group.

`!` The _policies_ define the _permissions_ of the users

`!` In AWS one uses __least privilige prinicple__, give only the permissions that a user needs

#### Hands on creating users and groups 

Do: Search "IAM" -> IAM Dashboard -> LHS :: "Users" // this is where one creates users for IAM  

__NOTE__: Region seelction is not  active here. It is _Global_ by default. A user in IAM is a global user.  

If you click on your name in upper right corner, you see _only your ID_ -> You have only ONE user.  
It is not adviced to use root account.  

`!` Creatig an _admin_ account for yourself is a good practive. 

Pick "Create a user", give name and give access to the console . Go with "I want to create an IAM user" as it is simpler. 
chose a password (I used the same as for AWS), and do not ask a user to change it when they log it. Create a group "admin" and chose _policy name_ as "AdministativeAccess" (the first one) and click "create", than add the user to the newly created group by clicking on it and go next.
__NOTE__: There are _tags_. These are _optional_, as allow to give _metadata_ to many resources.
Click 'create a user' and adfter, 'return to the user list'. There we have our user and our user group. Clicking on a group we can see a user and its permissions. 
__NOTE__: here a user has an _AdministrativeAccess_ but attached via "Group _Admin_", so the permissions are _inherited_ of the group. 

___Signing up with the created user__: 

__NOTE__ in the main IAM dashboard there is "AWS Account" section and "Sign-in URL for IAM users in this account". The 'url' can be customized by creating an account _alias_. I created "aws-vsevolod" alias.  
With this _alias_ the signing url can be _simplified_. 

Create a new "private" google chome window and copy paste there the sign-un URL. 

It prmpts to to "Sign in as IAM user". There a link "Sign in using root user email" will bring one back to the amin page, where one can choose to sign as as "Root user" or as a "IAM user", where in the latter we can enter the account _alias_. There "IAM user name" is the one we created, so "vosevolod" and the password is the one we created. 

__NOTE__: Now you are signed in as a "IAM user: vsevolod" with a _NEW_ account ID.   
__NOTE__ do not lose you account details  

### IAM Policies

Within a group _all_ users are given the same policy as the polycy of the group. 

For a _single_ user, there is an _inline policy_ that is attached to one user. 

If a user is a part of _multiple_ groups he iherits _policies_ from __All groups__. 

In AWS the policy is a `.json` document, as 

```python
{
    "version": 2012-10-17 # Version number with the data
    "id": "S3-account-permission", # id, as an identified (optional)
    "Statement": # multiple statmenets
    {
        "sid":"1", # a statement id (optional)
        "Effect": "Allow", #whethe the statmenet allows/denies the effect
        "Principal":[ # account/user/role to whch the policy applies
            "AWS": ["arn:aws:iam:123123:root"]
        ]
        "Action": "ec2:Describe",# list of api calls that would be either allowed ot denied based on the effect
        "Resources": [ # list of the resources to which the actions will be applied to
            "arn:aws:s3:mybucket/*"
        ]
    },
}
```

Sometimes there is also a condition for when the statmement should be applied or not (but it us optional)