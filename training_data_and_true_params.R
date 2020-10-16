library(purrr)
library(rstan)
library(bayesplot)
library(ggplot2)
library(mvtnorm)
library(parallel)
library(usethis)
library(reshape2)
library(ggplot2)

#seed = 234
seed = 250 #20 #252 #254
set.seed(seed)
L = 5
D = 200
N = 8

#create the true parameters
#Sigma_uu <- matrix(0, nrow = L, ncol = 1)
Sigma_uu = diag(L)
mu <- matrix(0, nrow = 1, ncol = L)
for (i in 1:L) {
    mu[1,i] <- rnorm(1,0,10)
}
alpha <- matrix(0, nrow = L, ncol = N)
for(i in 1:L){
    for(j in 1:N){
        alpha[i, j] <- rnorm(1, 0, 1)
    }
}
beta <- matrix(0, nrow = N, ncol = 1)
eta <- matrix(0, nrow = L, ncol = 1)
for (i in 1:N) {
    beta[i,1] <- rnorm(1,0,1)
}
for (i in 1:L) {
    eta[i,1] <- rnorm(1,0, 10)
}
tau <- matrix(rnorm(1,0,10), nrow = 1, ncol = 1)

set.seed(seed)
f_u <- function(mu, Sigma_uu){
    #Sigma_uu = diag(L)
    u <- matrix(0, nrow=D, ncol=L)
    for(i in 1:D){
        for(j in 1:L){
            u[i, j] <- rnorm(1, mu[1,j], sqrt(Sigma_uu[j,j]))
        }
    }
    return(list(u = u
                #, Sigma_uu = Sigma_uu
    ))
}
sim <- f_u(mu, Sigma_uu)
u_train <- sim$u
#Sigma_uu  <- sim$Sigma_uu

f_x <- function(u, Sigma_uu, alpha){
    linear_exp = u %*% alpha
    Sigma_xx = t(alpha) %*% Sigma_uu %*% alpha + diag(N)
    x <- matrix(0, nrow = D, ncol = N)
    for(i in 1:D){
        for(j in 1:N){
            x[i, j] <- rnorm(1, linear_exp[i,j],sqrt(Sigma_xx[j,j]))
        }
    }
    return(list(x = x, Sigma_xx = Sigma_xx))
}
sim_x <- f_x(u_train, Sigma_uu, alpha)
x_train <- sim_x$x
Sigma_xx <- sim_x$Sigma_xx

f_m <- function(x, Sigma_xx, beta){
    linear_exp = x %*% beta
    Sigma_mm = t(beta) %*% Sigma_xx %*% beta + 1
    m <- matrix(0, nrow = D, ncol = 1)
    for(i in 1:D){
        m[i, 1] <- rnorm(1, linear_exp[i,1],sqrt(Sigma_mm[1,1]))
    }
    return(list(m = m, Sigma_mm = Sigma_mm[1,1]))
}
sim_m <- f_m(x_train, Sigma_xx, beta)
m_train <- sim_m$m
Sigma_mm <- sim_m$Sigma_mm

f_y <- function(u, m, Sigma_uu, Sigma_mm, alpha, beta, eta, tau){
    linear_exp = m %*% tau + u %*% eta
    Sigma_yy = t(tau) %*% Sigma_mm %*% tau +
        2 * (t(eta) %*% Sigma_uu %*% alpha %*% beta %*% tau) +
        t(eta) %*% Sigma_uu %*% eta + 1
    y <- matrix(0, nrow = D, ncol = 1)
    for(i in 1:D){
        y[i, 1] <- rnorm(1, linear_exp[i,1],sqrt(Sigma_yy[1,1]))
    }
    return(list(y = y, Sigma_yy = Sigma_yy[1,1]))
}
sim_y <- f_y(u_train, m_train, Sigma_uu, Sigma_mm, alpha, beta, eta, tau)
y_train <- sim_y$y
Sigma_yy <- sim_y$Sigma_yy



get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}

plot_heatmap <- function(melted_cormat) {
    # Heatmap
    ggheatmap <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
        geom_tile(color = "white")+
        scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                             midpoint = 0, limit = c(-1,1), space = "Lab",
                             name="Pearson\nCorrelation") +
        theme_minimal()+
        theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                         size = 12, hjust = 1))+
        coord_fixed()

    ggheatmap <- ggheatmap +
        geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) #+
    theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank()#,
        #legend.justification = c(1, 0),
        #legend.position = c(0.6, 0.7),
        #legend.direction = "horizontal")+
        #guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
        #title.position = "top", title.hjust = 0.5)
    )
    return(ggheatmap)
}
