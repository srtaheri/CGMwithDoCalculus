//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
data {
  int L;
  int D;
  int N;
  matrix[D, N] x;
  matrix[D, 1] m;
  matrix[D, 1] y;
  matrix[1,N] x_1;
  matrix[1,N] x_2;
}
parameters {
  matrix[1, L] mu;
  matrix[L, N] alpha;
  matrix[N, 1] beta;
  matrix[L, 1] eta;
  matrix[1,1] tau;
  vector<lower=0>[L] u_scale;
  matrix[D, L] u;
}
transformed parameters {
  cov_matrix[L] sigma_uu;
  matrix[D, N] x_loc;
  matrix[D, 1] m_loc;
  matrix[D, 1] y_loc;
  vector[N] x_scale;
  vector[1] y_scale;
  vector[1] m_scale;
  matrix[N, N] sigma_xx;
  matrix[1,1] sigma_mm;
  matrix[1,1] sigma_yy;
  sigma_uu = diag_matrix(u_scale);
  sigma_xx = alpha'*sigma_uu*alpha + diag_matrix(rep_vector(1, N));
  x_scale = sqrt(diagonal(sigma_xx));
  sigma_mm = beta'*sigma_xx*beta + diag_matrix(rep_vector(1, 1));
  m_scale = sqrt(diagonal(sigma_mm));
  sigma_yy = tau'*sigma_mm*tau + 2*eta'*sigma_uu*alpha*beta*tau + eta'*sigma_uu*eta + diag_matrix(rep_vector(1, 1));
  y_scale = sqrt(diagonal(sigma_yy));
  for (i in 1:D){
    x_loc[i, ] = u[i, ] * alpha;
    m_loc[i, ] = x[i, ] * beta;
    y_loc[i, ] = m[i,] * tau + u[i,] * eta;
  }
}
// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += normal_lpdf(u_scale[1] | 1, 1);
  target += normal_lpdf(mu[1,] | 0, 10);
  target += normal_lpdf(tau[1,1] | 0, 10);
  for (j in 1:L){
    target += normal_lpdf(alpha[j, ] | 0, 1);
    target += normal_lpdf(eta[j, ] | 0, 10);
  }
  for (j in 1:N){
    target += normal_lpdf(beta[j, ] | 0, 1);
  }
  for (i in 1:D){
    target += normal_lpdf(u[i, ] | mu[1,], u_scale);      // likelihood
    target += normal_lpdf(x[i, ] | x_loc[i, ], x_scale);
    target += normal_lpdf(m[i, ] | m_loc[i, ], m_scale);
    target += normal_lpdf(y[i, ] | y_loc[i, ], y_scale);
  }
}
//generated quantities{
//  real y_do_x1;
//  real y_do_x2;
//  y_do_x1 = normal_rng((x_1 * beta * tau + mu * eta)[1,1], (tau' * tau + eta' * sigma_uu * eta)[1,1]+1);
//  y_do_x2 = normal_rng((x_2 * beta * tau + mu * eta)[1,1], (tau' * tau + eta' * sigma_uu * eta)[1,1]+1);
//}
