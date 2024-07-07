# Orbit Insight

## Description
Orbit Insight is a multidimensional orbit visualisation and analysis tool. 
The program has the following features:
  - Conversion of state vectors into Keplerian parameters
  - Accessing and analysing online satellite launch databases
  - Orbit visualisation using the matplotlib library

## Instructions and sample execution
Steps to run the code: execute Orbit_Insight.py.

The program is set up in a way where the user only needs to run the main function within
Orbit_Insight.py. The program will prompt the user for input and trigger appropriate function
calls automatically.

Below are two examples of possible prompt-response combinations based on the two available
options of the first prompt. Items in __bold__ represent user responses to program prompts.

### Example 1
You want to access a dataset and visualise the types of orbits of the satellites in
it. Then visualise a randomly chosen orbit.

Please follow the prompts:\
Option 1: Analyse satellite datasets and simulate existing orbit.\
Option 2: Enter state vectors to simulate specific orbit.\
Enter option number (1 or 2) (default value = 1): __1__

Select a dataset from the following options ('Debris' default):\
'Latest' - Compare orbits of all satellites launched in the recent 30 days.\
'Active' - Compare orbits of all active satellites.\
'Debris' - Compare orbits of all actively tracked debris and space objects.
__Debris__

Choose orbit l, m, or h: __m__

### Example 2
You want to convert your state vectors to Keplerian parameters and visualise the
corresponding orbit.

Please follow the prompts:\
Option 1: Analyse satellite datasets and simulate existing orbit.\
Option 2: Enter state vectors to simulate specific orbit.\
Enter option number (1 or 2) (default value = 1): __2__

Enter a position vector r = xi+yj+zk, or press Enter for default vectors (ISS position vector): __-
4552.7i-4431.9j+3478.4k__\
Enter a velocity vector v = xi+yj+zk, or press Enter for default vector (ISS velocity vector):
__2.3i+5.7j+4.2k__


