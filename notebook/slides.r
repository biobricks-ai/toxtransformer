# Load required libraries
library(ggplot2)
library(ggforce)

# Set the number of points
n <- 25

# Create a data frame with x and y coordinates from a standard normal distribution
df <- data.frame(x = 3*rnorm(n), y = 3*rnorm(n))

# Create a vector of radii from a uniform distribution
radius <- runif(n, 0.1, 1)

# Create a vector of colors
colors <- sample(colors(), n)

# Create a scatter plot with varying size and color
p <- ggplot(df, aes(x0 = x, y0 = y)) +
  geom_circle(aes(r = radius, color = colors, fill=colors), alpha = 0.6) +
  # scale_size(range = c(0.1, 5)) + 
  theme_minimal() +
  theme(legend.position = "none") +
  coord_fixed()  # To maintain aspect ratio

p + 
  geom_circle(aes(x0 = 0, y0 = 0, r = 3), color = "black", fill = NA, size = 1) +
  geom_circle(aes(x0 = 0, y0 = 0, r = 10), color = "black", fill = NA, size = 1) +
  geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black", fill = "#a300bc", size = 1)


# BAYES
# Load required libraries
library(ggplot2)

# Set up a grid of probability values
p <- seq(0, 1, length.out = 1000)

# Define the parameters of the beta prior
alpha_prior <- 2
beta_prior <- 2

# Define the observed data
successes <- 7
failures <- 3

# Compute the parameters of the beta posterior
alpha_posterior <- alpha_prior + successes
beta_posterior <- beta_prior + failures

# Compute the densities of the prior and posterior
prior <- dbeta(p, alpha_prior, beta_prior)
posterior <- dbeta(p, alpha_posterior, beta_posterior)

# Combine everything into a data frame
df <- data.frame(p = p, prior = prior, posterior = posterior)

# Create the plot
ggplot(df, aes(x = p)) +
  geom_line(aes(y = prior), color = "blue", linetype = "dashed") +
  geom_line(aes(y = posterior), color = "red") +
  labs(x = "Probability of Success", y = "Density", 
       title = "Bayesian Update: Beta-Binomial Model") +
  theme_minimal() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))  # To ensure the lines touch the x-axis

# CONTRASTIVE LEARNING ===========================================================
# Load required libraries
library(ggplot2)
library(gganimate)

# Define the initial positions of the points
df <- data.frame(
  i = 1:3,
  x = c(-1, 0, 1),
  y = c(0, 0, 0),
  color = c("red", "blue", "red")
)

g1 <- list(x=0, y=0, color="darkgreen")
g2 <- list(x=3, y=0, color="darkgreen")
r1 <- list(x=-1, y=0, color="darkred")

# Define a function to move points
# Create a data frame for each frame of the animation
n_frames = 10
df_list <- lapply(seq_len(n_frames), function(i) {
  r1 <<- list(x=r1$x-0.1, y=r1$y, color="darkred")
  g1 <<- list(x=g1$x+0.1, y=g1$y, color="darkgreen")
  g2 <<- list(x=g2$x-0.1, y=g2$y, color="darkgreen")
  data.frame(frame=rep(i,3), x=c(r1$x,g1$x,g2$x), y=rep(0,3), 
    color=c(r1$color,g1$color,g2$color))
})

# Bind the data frames together
df_anim <- dplyr::bind_rows(df_list)

# Create the initial plot
p <- ggplot(df_anim, aes(x = x, y = y, color = color)) +
  geom_point(size = 20) +
  theme_minimal() +
  theme(legend.position = "none") +
  coord_cartesian(xlim = c(-2, 4), ylim = c(-1, 1)) +
  labs(title = "Frame: {closest_state}")

# Create the animation
animation <- p +
  transition_states(frame, transition_length = 1, state_length = 1) +
  ease_aes('linear')

animate(animation, renderer = gifski_renderer("moving_points.gif"), duration = 2, fps = 10, width = 800, height = 600)
