from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan
from mrpPollLibrary import getMRPpoll
from mrpModel import mrpNestedModel2

#intOutcome = "ideo5_2016"
#question   = "affirmact_gen_2016"
#dichot     = None
#intPoll    = getMRPpoll("VSG16",intOutcome,None,[])
#intPoll.df["Success"] = 0
#intPoll.df.ix[intPoll.df[question].isin(dichot), 'Success'] = 1
#intPoll.uniq_survey_df.to_csv("ideologyMultinomial.csv",index=False)

intOutcome = "ideo5_2016"
question = "teapartsup_baseline"
dichot = None
drop = [6,8]
intPoll    = getMRPpoll("VSG16",intOutcome, dichot, drop)
intPoll.df["Success"] = intPoll.df[question]
intPoll.df = intPoll.df[~intPoll.df[question].isin(drop)]


#intPoll.df.ix[intPoll.df[question].isin(dichot), 'Success'] = 1
#intPoll.uniq_survey_df.to_csv("ideologyMultinomial.csv",index=False)


poll    = getMRPpoll("VSG16",question,dichot,drop,
                     additionalPredictors=[intOutcome, "newsint_2016"],
                     additionalPredictorsDrop={intOutcome : [], 
                                               "newsint_2016": [7] })
poll.df["Success"] = poll.df[question]
poll.df = poll.df[~poll.df[question].isin(drop)]


poll.uniq_survey_df.to_csv("affirmWIdeology.csv",index=False)

#polls = []
#for outcome in intPoll.outcome:
#    dft = intPoll.df[intPoll.df[intOutcome]==outcome]
#    uniq_survey_df["n"]           = uniq_survey_df['size']
#    uniq_survey_df["Success"]     = uniq_survey_df['sum']
#    polls.append(uniq_survey_df)
#
stateModel = mrpNestedModel2(intPoll,poll,1,intOutcome, infVar="newsint_2016")

pystan.misc.stan_rdump(stateModel.data, "data.R")
sm = pystan.StanModel(file='MRP.stan')
fit = sm.sampling(data=stateModel.data, iter=2000, chains=2,
                  n_jobs=2)
print(fit)

sums = fit.summary()

rhat = sums['summary'][:,9]
rhat = rhat[~np.isnan(rhat)]
plt.hist(rhat)

eta_infM = pd.Series(fit['eta_inf'].mean(0))
eta_infSE = pd.Series(fit['eta_inf'].std(0))

etaM = pd.Series(fit['eta'].mean(0))
etaSE = pd.Series(fit['eta'].std(0))



