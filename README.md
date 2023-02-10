# Historical Characteristics of Election Violence

#### Purpose:
This code aims to utilize machine learning (ML) to model characteristics of elections that correlate to election-related violence. 

#### Methodology:
Characteristics of global elections in history are used as independent input variables to train a Random Forest model to predict the level of election violence that occurred during the year leading up to and associated with a given election.

This analysis measures electoral violence in number of fatalities and classifies the level of electoral violence during an election cycle as one of three categories:
1. Non-fatal: no election-related fatalities recorded
2. Low-fatality: 1-3 election-related fatalities recorded
3. Mass-fatality: 4 or more election-related fatalities recorded

These categories are defined based upon a definition that 4 or more deaths constitutes a mass casualty event. ([Federal Bureau of Investigation](https://www.ojp.gov/ncjrs/virtual-library/abstracts/serial-murder-multi-disciplinary-perspectives-investigators))

The value of the machine learning model is to extract feature importances as a method of identifying which characteristics of elections are most informative in assessing the level of violence during an election period. Permutation importance is a specific tactic for obtaining feature importances. For this analysis, it quantifies the historical characteristics of elections that correlate to election violence by measuring which features are most influential on the ML model's ability to correctly classify the level of violence associated with an election.

#### Data Sources:
1. **Dataset of National Elections Across Democracy and Autocracy (NELDA)**
    - A historical dataset of the national elections for all independent countries from 1945-2020
    - Features Types: 
        - Election history of the country
        - Structure and quality of management of the election in question (e.g., whether opposition is allowed, delayed vote counting) 
        - Public perceptions of election fairness
        - The occurrence of protests
        - Economic and political state of the country (e.g., whether the country receives economic aid, impact of the election on US/international relations)
        - The presence of international monitors
    - Source: https://www.jstor.org/stable/23260172 or https://nelda.co/
    
    
2. **The Deadly Electoral Conflict Dataset (DECO)**
    - A georeferenced events dataset from the Uppsala Conflict Data Program (UCDP) that records incidents of electoral violence between 1989-2017 in which at least one election-related fatality occurred
    - Source: https://journals.sagepub.com/doi/full/10.1177/00220027211021620 or https://ucdp.uu.se/downloads/index.html#deco
