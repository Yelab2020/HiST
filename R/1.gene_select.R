library(sf)
library(sp)
library(spdep)
library(Seurat)
library(ggplot2)
library(magrittr)
library(tidyverse)

sample_list <- c()

moran_cutoff <- 0.4
data_path <- "./data/seurat_obj"
moran_csv_path <- './output/gene_select/moran/'
dir.create(moran_csv_path, showWarnings = FALSE, recursive = TRUE)
loc_pattern <- "\\.rds*"
seurat_obj_rds_paths <- list.files(data_path, pattern = loc_pattern, recursive = T, full.names = T)


#morani
moran_c <- function(TumorST, gene, tes_w_binary, tmp) {
  # moran(TumorST@assays$Spatial@data[gene,], tes_w_binary, length(tmp$geometry), Szero(tes_w_binary))[1]
  tes <- moran.test(TumorST@assays$Spatial@data[gene,],tes_w_binary, alternative="greater")
  Moran_I <- tes$estimate[1]
  p_value <- tes$p.value
  return(c(Moran_I, p_value))
}


for (i in 1:length(sample_list)) {
  sample_id <- sample_list[i]
  if (sample_id %in% sample_list[1:16]){
    next
  }
  TumorST <- readRDS(seurat_obj_rds_paths[i])
  genelist <- names(which(rowSums(TumorST@assays$Spatial@data) != 0))
  coords <- data.frame(
    x = TumorST@images$image@coordinates$col,
    y = max(TumorST@images$image@coordinates$row) - TumorST@images$image@coordinates$row
  )
  coords[,genelist]  <-  TumorST@assays$Spatial@data[genelist,]
  tmp <- SpatialPointsDataFrame(
    coords = coords[, c("x", "y")],
    data = coords
  )
  tmp <- sf::st_as_sf(tmp, coords = c("x", "y"))
  nb_15nn <- knn2nb(knearneigh(cbind(tmp$x, tmp$y), k=15))
  tes_w_binary <- nb2listw(nb_15nn, style="B")
  pboptions(do_eta = TRUE)
  svg <- do.call(rbind, pblapply(genelist, function(gene) {
    moran_c(TumorST, gene, tes_w_binary, tmp)
  }))
  svg_df <- as.data.frame(svg)
  colnames(svg_df) <- c('moranI','p_value')
  rownames(svg_df) <- genelist
  write.csv(svg_df, paste0(outdir,sample_id,'.csv'))
}


moran_csvs <- list.files(moran_csv_path,full.names = T)
#common genes in all samples
pb <- progress_bar$new(
  format = "[:bar] :percent ETA: :eta",
  total = length(sample_list)
)

for(sample_id in sample_list){
  loc_rds <- readRDS(paste0(data_path,sample_id,paste0('/',sample_id,'.rds.gz')))
  if(sample_id == sample_list[1]){
    all_genenames <- rownames(loc_rds@assays$Spatial@counts)
  }else{
    all_genenames <- all_genenames[all_genenames %in% rownames(loc_rds@assays$Spatial@counts)]
  }
  pb$tick()
}


#p value
for(i in seq_along(moran_csvs)){
  csv <- read.csv(moran_csvs[i])
  csv <- csv[which(csv$p_value<0.01),]
  csv <- csv[which(csv$X %in% all_genenames),c('X','moranI')]
  if(i==1){moran_df <- csv}else{
    moran_df <- merge(moran_df,csv,by = 'X')
  }
}
moran_df <- column_to_rownames(moran_df,var = 'X')
colnames(moran_df) <- sample_list

#distribution
ggplot(data.frame(moranI = unlist(moran_df)),aes(x= moranI)) + 
  geom_density()+
  theme_bw()+ ylab('Density')+
  geom_vline(xintercept = moran_cutoff,
             color="red",linetype="dashed")+
  annotate("text", x = moran_cutoff, y = max(unlist(moran_df)), label = sprintf("cutoff: %.2f", moran_cutoff),
           vjust = -1, hjust = -0.5, angle = 90, color = "red", size = 3)
ggsave('./output/gene_select/moranI_cutoff.pdf',width = 5,height = 5)


#median
medians <- apply(moran_df,1,median)
genelist <- rownames(moran_df)[which(medians > moran_cutoff)]
genelist_filtered <- genelist[grep('^RP',genelist,invert = T)]
genelist_filtered <- genelist_filtered[grep('^MT-',genelist_filtered,invert = T)]
genelist_filtered <- sort(genelist_filtered)

#cell lineage marker(for CRC and HCC)
lineage_df <- read_csv('./resource/MainTypesLineageMarker.csv')
lineage_markers <- unlist(lineage_df) %>% na.omit()

#save genelist
genelist_final <- unique(c(genelist_filtered,lineage_markers)) %>% sort()
write_csv(data.frame(genelist_final),
          file = './output/gene_select/SVG_list.txt',
          col_names = F)