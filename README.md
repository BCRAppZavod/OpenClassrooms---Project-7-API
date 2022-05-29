# OpenClassrooms---Project-7-API
API project: Predict loaner's solvency and computing portfolio financial performance


The main purpose of the API is to:

 - allow the user to make different portfolio scenarios
 - display a macro-analysis, i-e a portfolio global performance
 - display a micro-analysis, i-e a specific loaner profile based on its application and behavioral data

More specifically:
The API should be able to let the user choose between two hypotheses and try different kinds of portfolio:

 - the long-term interest rate (3%, 5%, or 7%) which represents the return on investment fok the bank.
 - the acceptance rate (0%, 10%, 20%, ..., to 100%) which represents the maximum decile in probabilities of default distribution, above which the loaners who have a higher risk of default will not reveive the credit.

Then, it should return the main financial indicators: the net loss, the cost of missed opportunities, the net gain, the mean loan, as well as the classification key-scores: AUROC, accuracy, the bad rate (weight of False Negative).

Finally, the API must allow the user to see for each potential loaner (by selecting the ID) its probability of default, its predicted status, and the graphical visualization of the most important features used by the model to compute default probabilities.

How does it work?
The API fetches a CSV file compressed with PICKLE which contains the loaners behavioral and application data (sex, number of children, the kind of work contract, annual wage, the amount of credit required, etc.).
Then it uses a Random Forest classifier to compute for each loaner in the dataset its probability of defaults. Both portfolio and loaner's profile are based on these probabilities returned by the classifier.
