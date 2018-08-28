data {
  int<lower = 1> nEffects_z;
  int<lower = 1> nCellPopulation;
  int indexes_Pop[nCellPopulation,nEffects_z];
  int N_Pop[nCellPopulation];
  int<lower = 1> nCellSample_z;
  int<lower = 2> nResponse_z;
  int response_z[nCellSample_z,nResponse_z];
  int indexes_z[nCellSample_z,nEffects_z];
  int<lower = 1> nCellSample;
  int N[nCellSample];
  int response[nCellSample];
  int<lower = 1> nAgeCat;
  int<lower = 1> nEduCat;
  int<lower = 1> nRaceCat;
  int<lower = 1> nGenderCat;
  int<lower = 1> nMarCat;
  int<lower = 1> nUSRCat;
  int<lower = 1> nIncCat;
  int<lower = 1> nideo5_2016;
  int<lower = 1> nnewsint_2016;
  int<lower = 1> nEffects;
  int indexes[nCellSample,nEffects];
}
parameters{
  real intercept_z_2;
  vector[nAgeCat]delta_AgeCat_z_2;
  real <lower=0> stdv_AgeCat_z_2;
  vector[nEduCat]delta_EduCat_z_2;
  real <lower=0> stdv_EduCat_z_2;
  vector[nRaceCat]delta_RaceCat_z_2;
  real <lower=0> stdv_RaceCat_z_2;
  vector[nGenderCat]delta_GenderCat_z_2;
  real <lower=0> stdv_GenderCat_z_2;
  vector[nMarCat]delta_MarCat_z_2;
  real <lower=0> stdv_MarCat_z_2;
  vector[nUSRCat]delta_USRCat_z_2;
  real <lower=0> stdv_USRCat_z_2;
  vector[nIncCat]delta_IncCat_z_2;
  real <lower=0> stdv_IncCat_z_2;
  real intercept_z_3;
  vector[nAgeCat]delta_AgeCat_z_3;
  real <lower=0> stdv_AgeCat_z_3;
  vector[nEduCat]delta_EduCat_z_3;
  real <lower=0> stdv_EduCat_z_3;
  vector[nRaceCat]delta_RaceCat_z_3;
  real <lower=0> stdv_RaceCat_z_3;
  vector[nGenderCat]delta_GenderCat_z_3;
  real <lower=0> stdv_GenderCat_z_3;
  vector[nMarCat]delta_MarCat_z_3;
  real <lower=0> stdv_MarCat_z_3;
  vector[nUSRCat]delta_USRCat_z_3;
  real <lower=0> stdv_USRCat_z_3;
  vector[nIncCat]delta_IncCat_z_3;
  real <lower=0> stdv_IncCat_z_3;
  real intercept_z_4;
  vector[nAgeCat]delta_AgeCat_z_4;
  real <lower=0> stdv_AgeCat_z_4;
  vector[nEduCat]delta_EduCat_z_4;
  real <lower=0> stdv_EduCat_z_4;
  vector[nRaceCat]delta_RaceCat_z_4;
  real <lower=0> stdv_RaceCat_z_4;
  vector[nGenderCat]delta_GenderCat_z_4;
  real <lower=0> stdv_GenderCat_z_4;
  vector[nMarCat]delta_MarCat_z_4;
  real <lower=0> stdv_MarCat_z_4;
  vector[nUSRCat]delta_USRCat_z_4;
  real <lower=0> stdv_USRCat_z_4;
  vector[nIncCat]delta_IncCat_z_4;
  real <lower=0> stdv_IncCat_z_4;
  real intercept_z_5;
  vector[nAgeCat]delta_AgeCat_z_5;
  real <lower=0> stdv_AgeCat_z_5;
  vector[nEduCat]delta_EduCat_z_5;
  real <lower=0> stdv_EduCat_z_5;
  vector[nRaceCat]delta_RaceCat_z_5;
  real <lower=0> stdv_RaceCat_z_5;
  vector[nGenderCat]delta_GenderCat_z_5;
  real <lower=0> stdv_GenderCat_z_5;
  vector[nMarCat]delta_MarCat_z_5;
  real <lower=0> stdv_MarCat_z_5;
  vector[nUSRCat]delta_USRCat_z_5;
  real <lower=0> stdv_USRCat_z_5;
  vector[nIncCat]delta_IncCat_z_5;
  real <lower=0> stdv_IncCat_z_5;
  vector[nAgeCat]delta_AgeCat;
  real <lower=0> stdv_AgeCat;
  vector[nEduCat]delta_EduCat;
  real <lower=0> stdv_EduCat;
  vector[nRaceCat]delta_RaceCat;
  real <lower=0> stdv_RaceCat;
  vector[nGenderCat]delta_GenderCat;
  real <lower=0> stdv_GenderCat;
  vector[nMarCat]delta_MarCat;
  real <lower=0> stdv_MarCat;
  vector[nUSRCat]delta_USRCat;
  real <lower=0> stdv_USRCat;
  vector[nIncCat]delta_IncCat;
  real <lower=0> stdv_IncCat;
  vector[nideo5_2016]delta_ideo5_2016;
  real <lower=0> stdv_ideo5_2016;
  vector[nAgeCat]delta_AgeCat_inf;
  real <lower=0> stdv_AgeCat_inf;
  vector[nEduCat]delta_EduCat_inf;
  real <lower=0> stdv_EduCat_inf;
  vector[nRaceCat]delta_RaceCat_inf;
  real <lower=0> stdv_RaceCat_inf;
  vector[nGenderCat]delta_GenderCat_inf;
  real <lower=0> stdv_GenderCat_inf;
  vector[nMarCat]delta_MarCat_inf;
  real <lower=0> stdv_MarCat_inf;
  vector[nUSRCat]delta_USRCat_inf;
  real <lower=0> stdv_USRCat_inf;
  vector[nIncCat]delta_IncCat_inf;
  real <lower=0> stdv_IncCat_inf;
  vector[nideo5_2016]delta_ideo5_2016_inf;
  real <lower=0> stdv_ideo5_2016_inf;
  vector[nnewsint_2016]delta_newsint_2016_inf;
  real <lower=0> stdv_newsint_2016_inf;
}
transformed parameters{
  vector[nAgeCat]a_AgeCat_z_2;
  vector[nEduCat]a_EduCat_z_2;
  vector[nRaceCat]a_RaceCat_z_2;
  vector[nGenderCat]a_GenderCat_z_2;
  vector[nMarCat]a_MarCat_z_2;
  vector[nUSRCat]a_USRCat_z_2;
  vector[nIncCat]a_IncCat_z_2;
  vector[nAgeCat]a_AgeCat_z_3;
  vector[nEduCat]a_EduCat_z_3;
  vector[nRaceCat]a_RaceCat_z_3;
  vector[nGenderCat]a_GenderCat_z_3;
  vector[nMarCat]a_MarCat_z_3;
  vector[nUSRCat]a_USRCat_z_3;
  vector[nIncCat]a_IncCat_z_3;
  vector[nAgeCat]a_AgeCat_z_4;
  vector[nEduCat]a_EduCat_z_4;
  vector[nRaceCat]a_RaceCat_z_4;
  vector[nGenderCat]a_GenderCat_z_4;
  vector[nMarCat]a_MarCat_z_4;
  vector[nUSRCat]a_USRCat_z_4;
  vector[nIncCat]a_IncCat_z_4;
  vector[nAgeCat]a_AgeCat_z_5;
  vector[nEduCat]a_EduCat_z_5;
  vector[nRaceCat]a_RaceCat_z_5;
  vector[nGenderCat]a_GenderCat_z_5;
  vector[nMarCat]a_MarCat_z_5;
  vector[nUSRCat]a_USRCat_z_5;
  vector[nIncCat]a_IncCat_z_5;
  matrix[nCellSample_z,nResponse_z] eta_z;
  vector[nCellSample_z] zeros;
  zeros = rep_vector(0,nCellSample_z);
  vector[nCellSample]eta;
  vector[nAgeCat]a_AgeCat;
  vector[nEduCat]a_EduCat;
  vector[nRaceCat]a_RaceCat;
  vector[nGenderCat]a_GenderCat;
  vector[nMarCat]a_MarCat;
  vector[nUSRCat]a_USRCat;
  vector[nIncCat]a_IncCat;
  vector[nideo5_2016]a_ideo5_2016;
  vector[nCellSample]eta_inf;
  vector[nAgeCat]a_AgeCat_inf;
  vector[nEduCat]a_EduCat_inf;
  vector[nRaceCat]a_RaceCat_inf;
  vector[nGenderCat]a_GenderCat_inf;
  vector[nMarCat]a_MarCat_inf;
  vector[nUSRCat]a_USRCat_inf;
  vector[nIncCat]a_IncCat_inf;
  vector[nideo5_2016]a_ideo5_2016_inf;
  vector[nnewsint_2016]a_newsint_2016_inf;
  eta_z[:,1] = zeros;
  a_AgeCat_z_2= stdv_AgeCat_z_2 * delta_AgeCat_z_2;
  a_EduCat_z_2= stdv_EduCat_z_2 * delta_EduCat_z_2;
  a_RaceCat_z_2= stdv_RaceCat_z_2 * delta_RaceCat_z_2;
  a_GenderCat_z_2= stdv_GenderCat_z_2 * delta_GenderCat_z_2;
  a_MarCat_z_2= stdv_MarCat_z_2 * delta_MarCat_z_2;
  a_USRCat_z_2= stdv_USRCat_z_2 * delta_USRCat_z_2;
  a_IncCat_z_2= stdv_IncCat_z_2 * delta_IncCat_z_2;
  eta_z[:,2] = a_AgeCat_z_2[indexes_z[:,1]] + a_EduCat_z_2[indexes_z[:,2]] + a_RaceCat_z_2[indexes_z[:,3]] + a_GenderCat_z_2[indexes_z[:,4]] + a_MarCat_z_2[indexes_z[:,5]] + a_USRCat_z_2[indexes_z[:,6]] + a_IncCat_z_2[indexes_z[:,7]];
  a_AgeCat_z_3= stdv_AgeCat_z_3 * delta_AgeCat_z_3;
  a_EduCat_z_3= stdv_EduCat_z_3 * delta_EduCat_z_3;
  a_RaceCat_z_3= stdv_RaceCat_z_3 * delta_RaceCat_z_3;
  a_GenderCat_z_3= stdv_GenderCat_z_3 * delta_GenderCat_z_3;
  a_MarCat_z_3= stdv_MarCat_z_3 * delta_MarCat_z_3;
  a_USRCat_z_3= stdv_USRCat_z_3 * delta_USRCat_z_3;
  a_IncCat_z_3= stdv_IncCat_z_3 * delta_IncCat_z_3;
  eta_z[:,3] = a_AgeCat_z_3[indexes_z[:,1]] + a_EduCat_z_3[indexes_z[:,2]] + a_RaceCat_z_3[indexes_z[:,3]] + a_GenderCat_z_3[indexes_z[:,4]] + a_MarCat_z_3[indexes_z[:,5]] + a_USRCat_z_3[indexes_z[:,6]] + a_IncCat_z_3[indexes_z[:,7]];
  a_AgeCat_z_4= stdv_AgeCat_z_4 * delta_AgeCat_z_4;
  a_EduCat_z_4= stdv_EduCat_z_4 * delta_EduCat_z_4;
  a_RaceCat_z_4= stdv_RaceCat_z_4 * delta_RaceCat_z_4;
  a_GenderCat_z_4= stdv_GenderCat_z_4 * delta_GenderCat_z_4;
  a_MarCat_z_4= stdv_MarCat_z_4 * delta_MarCat_z_4;
  a_USRCat_z_4= stdv_USRCat_z_4 * delta_USRCat_z_4;
  a_IncCat_z_4= stdv_IncCat_z_4 * delta_IncCat_z_4;
  eta_z[:,4] = a_AgeCat_z_4[indexes_z[:,1]] + a_EduCat_z_4[indexes_z[:,2]] + a_RaceCat_z_4[indexes_z[:,3]] + a_GenderCat_z_4[indexes_z[:,4]] + a_MarCat_z_4[indexes_z[:,5]] + a_USRCat_z_4[indexes_z[:,6]] + a_IncCat_z_4[indexes_z[:,7]];
  a_AgeCat_z_5= stdv_AgeCat_z_5 * delta_AgeCat_z_5;
  a_EduCat_z_5= stdv_EduCat_z_5 * delta_EduCat_z_5;
  a_RaceCat_z_5= stdv_RaceCat_z_5 * delta_RaceCat_z_5;
  a_GenderCat_z_5= stdv_GenderCat_z_5 * delta_GenderCat_z_5;
  a_MarCat_z_5= stdv_MarCat_z_5 * delta_MarCat_z_5;
  a_USRCat_z_5= stdv_USRCat_z_5 * delta_USRCat_z_5;
  a_IncCat_z_5= stdv_IncCat_z_5 * delta_IncCat_z_5;
  eta_z[:,5] = a_AgeCat_z_5[indexes_z[:,1]] + a_EduCat_z_5[indexes_z[:,2]] + a_RaceCat_z_5[indexes_z[:,3]] + a_GenderCat_z_5[indexes_z[:,4]] + a_MarCat_z_5[indexes_z[:,5]] + a_USRCat_z_5[indexes_z[:,6]] + a_IncCat_z_5[indexes_z[:,7]];
  a_AgeCat= stdv_AgeCat * delta_AgeCat;
  a_EduCat= stdv_EduCat * delta_EduCat;
  a_RaceCat= stdv_RaceCat * delta_RaceCat;
  a_GenderCat= stdv_GenderCat * delta_GenderCat;
  a_MarCat= stdv_MarCat * delta_MarCat;
  a_USRCat= stdv_USRCat * delta_USRCat;
  a_IncCat= stdv_IncCat * delta_IncCat;
  a_ideo5_2016= stdv_ideo5_2016 * delta_ideo5_2016;
  eta = a_AgeCat[indexes[:,1]] + a_EduCat[indexes[:,2]] + a_RaceCat[indexes[:,3]] + a_GenderCat[indexes[:,4]] + a_MarCat[indexes[:,5]] + a_USRCat[indexes[:,6]] + a_IncCat[indexes[:,7]] + a_ideo5_2016[indexes[:,8]];
  a_AgeCat_inf= stdv_AgeCat_inf * delta_AgeCat_inf;
  a_EduCat_inf= stdv_EduCat_inf * delta_EduCat_inf;
  a_RaceCat_inf= stdv_RaceCat_inf * delta_RaceCat_inf;
  a_GenderCat_inf= stdv_GenderCat_inf * delta_GenderCat_inf;
  a_MarCat_inf= stdv_MarCat_inf * delta_MarCat_inf;
  a_USRCat_inf= stdv_USRCat_inf * delta_USRCat_inf;
  a_IncCat_inf= stdv_IncCat_inf * delta_IncCat_inf;
  a_ideo5_2016_inf= stdv_ideo5_2016_inf * delta_ideo5_2016_inf;
  a_newsint_2016_inf= stdv_newsint_2016_inf * delta_newsint_2016_inf;
  eta_inf = a_AgeCat_inf[indexes[:,1]] + a_EduCat_inf[indexes[:,2]] + a_RaceCat_inf[indexes[:,3]] + a_GenderCat_inf[indexes[:,4]] + a_MarCat_inf[indexes[:,5]] + a_USRCat_inf[indexes[:,6]] + a_IncCat_inf[indexes[:,7]] + a_ideo5_2016_inf[indexes[:,8]] + a_newsint_2016_inf[indexes[:,9]];
}
model {
  intercept_z_2 ~ normal(0,100);
  delta_AgeCat_z_2 ~ normal(0,1);
  stdv_AgeCat_z_2 ~ normal(0,1);
  delta_EduCat_z_2 ~ normal(0,1);
  stdv_EduCat_z_2 ~ normal(0,1);
  delta_RaceCat_z_2 ~ normal(0,1);
  stdv_RaceCat_z_2 ~ normal(0,1);
  delta_GenderCat_z_2 ~ normal(0,1);
  stdv_GenderCat_z_2 ~ normal(0,1);
  delta_MarCat_z_2 ~ normal(0,1);
  stdv_MarCat_z_2 ~ normal(0,1);
  delta_USRCat_z_2 ~ normal(0,1);
  stdv_USRCat_z_2 ~ normal(0,1);
  delta_IncCat_z_2 ~ normal(0,1);
  stdv_IncCat_z_2 ~ normal(0,1);
  intercept_z_3 ~ normal(0,100);
  delta_AgeCat_z_3 ~ normal(0,1);
  stdv_AgeCat_z_3 ~ normal(0,1);
  delta_EduCat_z_3 ~ normal(0,1);
  stdv_EduCat_z_3 ~ normal(0,1);
  delta_RaceCat_z_3 ~ normal(0,1);
  stdv_RaceCat_z_3 ~ normal(0,1);
  delta_GenderCat_z_3 ~ normal(0,1);
  stdv_GenderCat_z_3 ~ normal(0,1);
  delta_MarCat_z_3 ~ normal(0,1);
  stdv_MarCat_z_3 ~ normal(0,1);
  delta_USRCat_z_3 ~ normal(0,1);
  stdv_USRCat_z_3 ~ normal(0,1);
  delta_IncCat_z_3 ~ normal(0,1);
  stdv_IncCat_z_3 ~ normal(0,1);
  intercept_z_4 ~ normal(0,100);
  delta_AgeCat_z_4 ~ normal(0,1);
  stdv_AgeCat_z_4 ~ normal(0,1);
  delta_EduCat_z_4 ~ normal(0,1);
  stdv_EduCat_z_4 ~ normal(0,1);
  delta_RaceCat_z_4 ~ normal(0,1);
  stdv_RaceCat_z_4 ~ normal(0,1);
  delta_GenderCat_z_4 ~ normal(0,1);
  stdv_GenderCat_z_4 ~ normal(0,1);
  delta_MarCat_z_4 ~ normal(0,1);
  stdv_MarCat_z_4 ~ normal(0,1);
  delta_USRCat_z_4 ~ normal(0,1);
  stdv_USRCat_z_4 ~ normal(0,1);
  delta_IncCat_z_4 ~ normal(0,1);
  stdv_IncCat_z_4 ~ normal(0,1);
  intercept_z_5 ~ normal(0,100);
  delta_AgeCat_z_5 ~ normal(0,1);
  stdv_AgeCat_z_5 ~ normal(0,1);
  delta_EduCat_z_5 ~ normal(0,1);
  stdv_EduCat_z_5 ~ normal(0,1);
  delta_RaceCat_z_5 ~ normal(0,1);
  stdv_RaceCat_z_5 ~ normal(0,1);
  delta_GenderCat_z_5 ~ normal(0,1);
  stdv_GenderCat_z_5 ~ normal(0,1);
  delta_MarCat_z_5 ~ normal(0,1);
  stdv_MarCat_z_5 ~ normal(0,1);
  delta_USRCat_z_5 ~ normal(0,1);
  stdv_USRCat_z_5 ~ normal(0,1);
  delta_IncCat_z_5 ~ normal(0,1);
  stdv_IncCat_z_5 ~ normal(0,1);
  intercept ~ normal(0,100);
  delta_AgeCat ~ normal(0,1);
  stdv_AgeCat ~ normal(0,1);
  delta_EduCat ~ normal(0,1);
  stdv_EduCat ~ normal(0,1);
  delta_RaceCat ~ normal(0,1);
  stdv_RaceCat ~ normal(0,1);
  delta_GenderCat ~ normal(0,1);
  stdv_GenderCat ~ normal(0,1);
  delta_MarCat ~ normal(0,1);
  stdv_MarCat ~ normal(0,1);
  delta_USRCat ~ normal(0,1);
  stdv_USRCat ~ normal(0,1);
  delta_IncCat ~ normal(0,1);
  stdv_IncCat ~ normal(0,1);
  for(n in 1:nCellSample_z)
    response_z[n,:]   ~ multinomial(softmax(to_vector(eta_z[n,:])));
  for(n in 1:nCellSample)
    response[n]   ~ binomial(N[n],inv_logit(eta[n]));
}
generated quantities {
  simplex[nResponse_z] probs;
  vector[nResponse_z] etaTemp_z;
  vector[nResponse_z] etaTemp;
  int countsTemp[nResponse_z];
  int totalYes;
  int totalN;
  real totalPct;
  totalYes=0;
  totalN=0;
  etaTemp_z[1]=0;
  for(i in 1:nCellPopulation){
    etaTemp_z[2] = a_AgeCat_z_2[indexes_Pop[i,1]] + a_EduCat_z_2[indexes_Pop[i,2]] + a_RaceCat_z_2[indexes_Pop[i,3]] + a_GenderCat_z_2[indexes_Pop[i,4]] + a_MarCat_z_2[indexes_Pop[i,5]] + a_USRCat_z_2[indexes_Pop[i,6]] + a_IncCat_z_2[indexes_Pop[i,7]];
    etaTemp_z[3] = a_AgeCat_z_3[indexes_Pop[i,1]] + a_EduCat_z_3[indexes_Pop[i,2]] + a_RaceCat_z_3[indexes_Pop[i,3]] + a_GenderCat_z_3[indexes_Pop[i,4]] + a_MarCat_z_3[indexes_Pop[i,5]] + a_USRCat_z_3[indexes_Pop[i,6]] + a_IncCat_z_3[indexes_Pop[i,7]];
    etaTemp_z[4] = a_AgeCat_z_4[indexes_Pop[i,1]] + a_EduCat_z_4[indexes_Pop[i,2]] + a_RaceCat_z_4[indexes_Pop[i,3]] + a_GenderCat_z_4[indexes_Pop[i,4]] + a_MarCat_z_4[indexes_Pop[i,5]] + a_USRCat_z_4[indexes_Pop[i,6]] + a_IncCat_z_4[indexes_Pop[i,7]];
    etaTemp_z[5] = a_AgeCat_z_5[indexes_Pop[i,1]] + a_EduCat_z_5[indexes_Pop[i,2]] + a_RaceCat_z_5[indexes_Pop[i,3]] + a_GenderCat_z_5[indexes_Pop[i,4]] + a_MarCat_z_5[indexes_Pop[i,5]] + a_USRCat_z_5[indexes_Pop[i,6]] + a_IncCat_z_5[indexes_Pop[i,7]];
    probs = softmax(etaTemp_z);
    countsTemp = multinomial_rng(probs,N_Pop[i]);
    totalN += N_Pop[i];
    etaTemp[1] = a_AgeCat[indexes_Pop[i,1]] + a_EduCat[indexes_Pop[i,2]] + a_RaceCat[indexes_Pop[i,3]] + a_GenderCat[indexes_Pop[i,4]] + a_MarCat[indexes_Pop[i,5]] + a_USRCat[indexes_Pop[i,6]] + a_IncCat[indexes_Pop[i,7]] + a_ideo5_2016[1];
    etaTemp[2] = a_AgeCat[indexes_Pop[i,1]] + a_EduCat[indexes_Pop[i,2]] + a_RaceCat[indexes_Pop[i,3]] + a_GenderCat[indexes_Pop[i,4]] + a_MarCat[indexes_Pop[i,5]] + a_USRCat[indexes_Pop[i,6]] + a_IncCat[indexes_Pop[i,7]] + a_ideo5_2016[2];
    etaTemp[3] = a_AgeCat[indexes_Pop[i,1]] + a_EduCat[indexes_Pop[i,2]] + a_RaceCat[indexes_Pop[i,3]] + a_GenderCat[indexes_Pop[i,4]] + a_MarCat[indexes_Pop[i,5]] + a_USRCat[indexes_Pop[i,6]] + a_IncCat[indexes_Pop[i,7]] + a_ideo5_2016[3];
    etaTemp[4] = a_AgeCat[indexes_Pop[i,1]] + a_EduCat[indexes_Pop[i,2]] + a_RaceCat[indexes_Pop[i,3]] + a_GenderCat[indexes_Pop[i,4]] + a_MarCat[indexes_Pop[i,5]] + a_USRCat[indexes_Pop[i,6]] + a_IncCat[indexes_Pop[i,7]] + a_ideo5_2016[4];
    etaTemp[5] = a_AgeCat[indexes_Pop[i,1]] + a_EduCat[indexes_Pop[i,2]] + a_RaceCat[indexes_Pop[i,3]] + a_GenderCat[indexes_Pop[i,4]] + a_MarCat[indexes_Pop[i,5]] + a_USRCat[indexes_Pop[i,6]] + a_IncCat[indexes_Pop[i,7]] + a_ideo5_2016[5];
    for(j in 1:nResponse_z){
      totalYes += binomial_rng(countsTemp[j],inv_logit(etaTemp[j]));
    }
  }
  totalPct = 100.*totalYes/totalN;
}
