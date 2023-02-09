# Historical Characteristics of Election Violence

#### Purpose:
This code aims to utilize machine learning to model characteristics of elections that correlate to election-related violence. 

#### Methdology:
Characteristics of global elections in history are used as independent input variables to train a Random Forest model to predict the level of election-related fatalities recorded in the one year preceeding the date of a given election.

This analysis measures electoral violence by number of fatalities and classifies the level of electoral violence during an election cycle as one of three categories:
1. Non-fatal: no election-related fatalities recorded
2. Low-fatality: 1-3 election-related fatalities recorded
3. Mass-fatality: 4 or more election-related fatalities recorded

The value of the machine learning model is extract feature importances as a method to identify which characteristics of elections are most informative in assessing the level of violence during an election period. Permutation importance is used to provide of list of historical characteristics of elections that are most influential on the ML model's ability to predict the level of election violence.
