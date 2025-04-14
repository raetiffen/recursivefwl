// Using "alg time.dta"

// Figure 2 - Iterated FWL times
regress time ibn.covariates if algorithm == 1, noconstant
coefplot, recast(connected) vertical yline(0)

// Figure 3 - Recursive FWL decompositions times
regress time ibn.covariates if algorithm == 2, noconstant
coefplot, recast(connected) vertical yline(0)