outdir <- fs::dir_create("notebook/output")
outpath <- purrr::partial(fs::path,outdir)

# KNN RESULTS ===================================================================
pacman::p_load(tidyverse, shiny)
df <- readr::read_csv("metrics/knn.csv") |> pivot_longer(cols=-c(fingerprint,assay))
df |> filter(name=="bac") |> group_by(fingerprint) |> 
  summarize(meanAUC=mean(value),medauc=median(value),nassays=n()) |>
  arrange(desc(meanAUC))

# HISTOGRAM OF MORGAN AUC VALUES ================================================
df <- readr::read_csv("metrics/knnpvae.csv") |> pivot_longer(cols=-c(fingerprint,assay))
pdf <- df |> filter(name=="auc", fingerprint=="morgan_pred") 
# httpgd::hgd()
ggplot(pdf, aes(x=value)) + geom_density() + 
  scale_y_continuous(expand = c(0,0.1)) + 
  xlab("AUC") + ylab("ASSAYS") + theme_bw()
ggsave(outpath("morgan_auc_hist.png"), width=5, height=5)

# Histogram of Morgan and predvae AUC values ====================================
df <- readr::read_csv("metrics/knnpvae.csv") |> pivot_longer(cols=-c(fingerprint,assay))
df$fingerprint = ifelse(df$fingerprint=="morgan_pred", "morgan", "vae")
pdf <- df |> filter(name=="auc") 
httpgd::hgd()
ggplot(pdf, aes(x=value,fill=fingerprint)) + geom_density(alpha=0.33) + 
  scale_y_continuous(expand = c(0,0.1)) + 
  xlab("AUC") + ylab("ASSAYS") + theme_bw()
ggsave(outpath("morgan_auc_hist.png"), width=5, height=5)

# Histogram of Morgan, Predvae, and vaecontrast AUC values =======================
df1 <- readr::read_csv("metrics/knnpvae.csv") |> select(method=fingerprint, auc)
df1$method = ifelse(df1$method=="morgan_pred", "morgan", "vae")

df2 <- readr::read_csv("metrics/vaecontrast.csv") |> mutate(method="vaecontrast")
df2 <- df2 |> select(method, auc)

pdf <- bind_rows(df1, df2)
httpgd::hgd()
ggplot(pdf, aes(x=auc,fill=method)) + geom_density(alpha=0.33) + 
  scale_y_continuous(expand = c(0,0.1)) + 
  xlab("AUC") + ylab("ASSAYS") + theme_bw() + 
  theme(legend.position="bottom")
ggsave("notebook/output/vaecontrast.png", width=5, height=5)
# AUC FINGERPRINT COMPARISON======================================================
compdf <- df |> filter(name=="auc") |> select(assay, fingerprint, auc=value)
compdf <- compdf |> pivot_wider(names_from=fingerprint, values_from=auc)
compdf <- compdf |> mutate(diff = biosim_pred - morgan_pred)

pdf <- df |> filter(name=="auc") |> select(assay, fingerprint, auc=value)
pdf <- pdf |> mutate(assay = reorder(assay, -auc, FUN=mean),
                     fingerprint = reorder(fingerprint, -auc, FUN=mean))

httpgd::hgd()
ggplot(pdf, aes(x=assay, y=fingerprint, fill=auc)) + 
  geom_tile(color="white") +
  scale_fill_gradient(low="#7e0404", high="#08b6b6") +
  theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1)) +
  labs(title="Assay Performance by Fingerprint", x="Assay", y="Fingerprint", fill="AUC") +
  coord_fixed()
