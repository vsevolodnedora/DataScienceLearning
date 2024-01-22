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

__NOTE__ in the main IAM dashboard there is "AWS Account" section and "Sign-in URL for IAM users in this account". The 'url' can be customized by creating an account _alias_. I created `aws-vsevolod` alias.  
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
        "sid":"1", # a statement id -- identifier for a statmenet (optional)
        "Effect": "Allow", # whether to allow allows/denies the effect
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

__NOTE__: for the the exam it is crucual to know 
- __Effect__
- __Principle__ 
- __Action__
- __Resource__

Sometimes there is also a condition for when the statmement should be applied or not (but it us optional)

If you log in as a root, you can remove access rights to users. Specifically, you can remove the newly created user from _admin_ group, and if you log in with your user, you will see that the user has _lost_ the ability to view the IAM dashboard  
[ Access denied ]

Now we want to add _policies_ to the user instead of adding him back to the group

Go: IAM -> Users -> this-user -> Add permissions (directly) -> Attach Policy Directly -> 'IAMReadOnlyAccess'

And now the user can see the IAM, however he _cannot_ read groups or assign policies. 

Then we created a new group with this user and we added this user back to the admin group so now the user has three policies
- AdministatorAccess
- AlexeForBusiness
- IAMReadOnlyAccess

And it shows via what group a user is added to a given policy.

Than you can click on a _permission_ and go the new page and there click on _JSON_, that looks like 

```python
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "*", # 
            "Resource": "*"
        }
    ]
}
```

where '*' stands for _everything_ aka __administrative access__. 

Another poloty, e.g., 'IAMReadOnly...' has thefollowing _JSON_:

```python
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:GenerateCredentialReport",
                "iam:GenerateServiceLastAccessedDetails",
                "iam:Get*", # anythong that starts with get is allowd
                "iam:List*",
                "iam:SimulateCustomPolicy",
                "iam:SimulatePrincipalPolicy"
            ],
            "Resource": "*"
        }
    ]
}
```

You can also see that it has an _Acess level_ as Full: 'List Limited: Read'. 

Clicking on a policy gives the _Fill list of Allowed API calls_

- We can also _create_ our own policy. We created the follwoing:

```python
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "iam:ListUsers",
                "iam:GetUser"
            ],
            "Resource": "*"
        }
    ]
}
```
Then, we deleted group and policy


### Defending User

- IAM - Password Policy
    - Strong password - better sequrity
- In AWS there are 
    - Spec. length
    - Spec. characters
        - upper/lower ... 
    - Allow users to change password / require users to change it regularly
    - Prevent password re-use

- Multi Factor Authentication - MFA (recommended)

It is a good practice to protext the _root  user_ and all _IAM users_  

MFA is a combination of 2 factors, e.g., a password and a device 

__MFA device options in AWS:__
- _Virtual MFA device_ (e.g., google authenticator (one phone only))
- _Authy_ (multy-device) (Support for multiple tokens on a single device)
    - This One MFA device supports _many_ users
- __Universal 2nd Factor (U2F) Security Key__ (this is a physical device, e.g., a flash drive), provided by a third partner
- __Hardware key Fob MFA Device__ (also provided by 3rd party)
- __Hardware Key Fob MFA Device for AWS__  GowCloud (US)


In practive we implement this is following
Go to 'account settings' -> 'password policy' -> edit -> custom

Setting an MFA for root account: go to the account upper right, and set there 'Assign MFA'; Follow the instructions. Choose an app that you want, e.g. Twollio 


### How can users access AWS

- __AWS managment console__(protexted by password and MFA)
- __AWS Command Line Interface__ (SLI) (protected by access keys)
- __AWS software Developer Kit (SDK)__ - for code; protected by access keys; used for APIs when used within applications

Access Kys are generated through _AWS Console_  
Users are responsible for access keys (they are private, not to be shared)

#### AWS CLI 

- A tool that allows to interact with AWS services using command-line shell
- Direct access to the public APIs or AWS services (use as `aws s3 ls s3://cpp-mybucket`)
- Develop scrips to manage resources
- Alterantive to using AWS Managment Console

#### AWS SDK 

- Language specific APIs (set of libraries)
- Enagles access and managment of AWS serveses _programmatically_ 
- Embedded within an application
- Supports Python, C++...
- Mobal SDK
- IoT Devices SDK ... 

I Istalled CLI following [link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-version.html)

Next: creating _access keys_: 
Go to IAM -> Users -> Sequrity credentials -> create an access key 

Go to the shell :: Configure
```bash
$ aws configure
$ AWS Access Key ID [None] < 1234567489
$ AWS Secret Access Key [None] < 123456789123456789
$ Default region name [None]: eu-central-1
$ Default output format [None]:
```

Go to the shell :: Explore 

```bash
$ aws iam list-users
```

__NOTE__: managment concil and CLI provide similar options


### AWS CloudShell 

Alternative to using local CLI

There is full repository there so we can create a text file there as 

```bash
$ echo "test" > demo.txt
```
and these files will _persisit_