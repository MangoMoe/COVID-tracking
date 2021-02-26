## Note

I did not make this tool nor did I help with the paper. This fork is for my sensitivity analysis of the tool itself to the input parameters in order to evaluate its effectiveness in normal and extreme cases.

# COVID-19 Outbreak Detection Tool

## About this project
[Main App Here](https://analytics-modeling.shinyapps.io/outbreakdetection/)

The COVID-19 Outbreak Detection Tool detects recent COVID-19 outbreaks in U.S. counties. The tool leverages machine learning to predict how fast an outbreak could spread at the county level by estimating the doubling time of COVID-19 cases. It accounts for reported COVID-19 cases and deaths, face mask mandates, social distancing policies, the CDC’s Social Vulnerability Index, changes in tests performed and rate of positive tests. The tool offers an interactive map and a data explorer allowing users to filter and rearrange counties based on predicted trends, which get updated at least once per week.

![alt text](https://github.com/Runespear/COVID-tracking/blob/master/OutbreakNY_1016.png?raw=true)

## Repository
Data files used for the analysis are included in the ```data``` folder.  
Source code for the backend is located at the ```src``` folder.  
The backend workflow can be found in the ```WorkFlow.txt``` document.  

## Reference and Methodology
To reference, please cite our paper [Estimating County-Level COVID-19 Exponential Growth Rates Using Generalized Random Forests
](https://arxiv.org/abs/2011.01219).

