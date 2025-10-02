pacman::p_load(ggplot2, readr, dplyr, scales, crayon)

args <- commandArgs(trailingOnly = TRUE)
batch_skip = if(length(args) > 0) as.integer(args[1]) else 0

draw_plot <- function(last_scheduler_length=0){
  # Read the first three columns of the metrics file
  data <- read_tsv('metrics/multitask_loss.tsv', col_names = c('type', 'batch', 'loss', 'lr'), skip =1, show_col_types = FALSE) 
  data <- data |> mutate(type = ifelse(type=="scheduler","sched",type))
  scheduler_length = length(data |> filter(type == "sched") |> pull(loss))
  if(scheduler_length == last_scheduler_length){
    return(last_scheduler_length)
  }
  data <- data |> mutate(batch = row_number()) 
  data <- data |> filter(batch > batch_skip)

  print_losses <- function(type){
    tl <- round(data |> filter(type == .env$type) |> pull(loss) |> tail(), 6)
    mn <- min(tl)
    msg <- purrr::map_chr(tl, ~ifelse(.x <= mn, crayon::green(.x), crayon::yellow(.x)))
    cat(type,"\t", msg, '\n')
  }

  epoch = data |> tail(1) |> pull(batch)
  last_lr = data |> tail(1) |> pull(lr)
  
  cat("Epoch:", epoch, "Last learning rate:", last_lr, "\n")
  print_losses("train")
  print_losses("sched")
  if(nrow(data |> filter(type =="eval")) > 0){ print_losses("eval") }
  
  data <- data #|> filter(type != "train")
  # Create the plot with a black theme and log scale for the y-axis
  plot <- ggplot(data, aes(x = batch, y=loss, col=type)) + 
    geom_point(aes(color = type),alpha=0.8, size=2) +
    # geom_smooth(size=1, alpha=0.5, col="red") +
    labs(x = 'Iteration', y = 'Loss', title = 'VAE Losses Over Iterations', color = "Metric") +
    # scale_y_continuous(limits = c(min(data$loss),max(data$loss))) +
    theme_minimal(base_size = 16) +
    theme(
      text = element_text(color = "white"),
      plot.background = element_rect(fill = "black"),
      panel.background = element_rect(fill = "black"),
      legend.background = element_rect(fill = "black", color = "black"),
      legend.position = "bottom",
      legend.title.align = 0.5,
      plot.title = element_text(color = "white"),
      axis.title = element_text(color = "white"),
      axis.text = element_text(color = "white"),
      axis.line = element_line(color = "#533e3e"),
      panel.grid.major = element_line(color = "#533e3e"),
      panel.grid.minor = element_line(color = "#533e3e")
    ) +
    guides(color = guide_legend(title = "Metrics")) + 
    facet_wrap(~type, scales="free_y")

  # Create the directory and save the plot
  dir.create('notebook/plots', recursive = TRUE, showWarnings = FALSE)
  # ggsave('notebook/plots/loss.svg', plot = plot, width = 12, height = 7, dpi = 300)
  ggsave('notebook/plots/loss.png', plot = plot, width = 12, height = 7, dpi = 300)
  return(scheduler_length)
}

last_scheduler_length = 0
while(TRUE){
  last_scheduler_length = draw_plot(last_scheduler_length)
  Sys.sleep(1)
}
