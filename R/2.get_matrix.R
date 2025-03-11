library(Seurat)
library(magrittr)
library(tidyverse)

#get geneMatrix
data_path <- "./data/seurat_obj"
rds_count_path <- './output/geneMatrix/count/'
rds_normedgene_path <- './output/geneMatrix/normed/'
csv_cnt_path <- './output/geneMatrix/cnt_csv/'
rds_mask_path <- './output/mask_rds/'
tissue_positions_path <- './output/tissue_positions_list/'
scale_factor_path <- './output/scale_factor/'
dir.create(rds_count_path,recursive = T,showWarnings = F)
dir.create(rds_normedgene_path,recursive = T,showWarnings = F)
dir.create(csv_cnt_path,recursive = T,showWarnings = F)
dir.create(rds_mask_path,recursive = T,showWarnings = F)
dir.create(tissue_positions_path,recursive = T,showWarnings = F)
dir.create(scale_factor_path,recursive = T,showWarnings = F)

loc_pattern <- "\\.rds\\.gz$"
seurat_obj_rds_paths <- list.files(data_path, pattern = loc_pattern, recursive = T, full.names = T)
sample_list <- gsub(loc_pattern, "", basename(seurat_obj_rds_paths))

genelist <- read.csv('./resource/CRC_SVG346_list.txt',header = F)$V1

create_geneMatrix <- function(input) {
  zero_array <- array(0, dim = c(80, 64, ncol(input)))
  
  rownames_split <- strsplit(rownames(input), split = '-')
  rows <- as.numeric(sapply(rownames_split, function(x) as.numeric(x[1])))
  cols <- as.numeric(sapply(rownames_split, function(x) as.numeric(x[2])))
  col_adjust <- ifelse(rows %% 2 == 0, cols / 2 + 1, (cols + 1) / 2)
  
  for (j in 1:ncol(input)) {
    zero_array[cbind(rows+1+2, col_adjust, rep(j,nrow(input)))] <- input[, j]#+1 means 0->1 ;  +2 means 76->80
  }
  
  return(zero_array)
}

create_spot_id_df <- function(posi_sorted){
  spot_id_df <- data.frame(matrix(NA, nrow = 78, ncol = 64))
  for(i in 1:nrow(posi_sorted)){
    if(posi_sorted$row[i]%%2==0){
      spot_id_df[posi_sorted$row[i]+1,posi_sorted$col[i]/2+1] <- posi_sorted$barcode[i]
    }else{
      spot_id_df[posi_sorted$row[i]+1,(posi_sorted$col[i]+1)/2] <- posi_sorted$barcode[i]
    }
  }
  return(spot_id_df)
}

create_mask <- function(type,spot_df){
  zero_mask <- matrix(0, nrow = 80, ncol = 64)
  for(i in 1:78){
    for(j in 1:64){
      if(!is.na(spot_df[i, j]) && spot_df[i,j]==type){
        zero_mask[i+1,j] <- 1
      }
    }
  }
  return(zero_mask)
}


for(i in 1:length(sample_list)){
  sample_id <- sample_list[i]
  message(paste0(sample_id,' Start process'))
  loc_rds <- readRDS(seurat_obj_rds_paths[i])
  posi_csv <- loc_rds@images$image@coordinates
  posi_csv <- tibble::rownames_to_column(posi_csv,var = 'barcode')
  write.csv(posi_csv,file = paste0(tissue_positions_path,sample_id,'.csv'),row.names = FALSE)

  scale.factors <- loc_rds@images$image@scale.factors
  scale_factors_df <- data.frame(spot = scale.factors$spot,
                                 fiducial = scale.factors$fiducial,
                                 hires = scale.factors$hires,
                                 lowres = scale.factors$lowres)
  write.csv(scale_factors_df,file = paste0(scale_factor_path,sample_id,'.csv'),row.names = F)

  posi_sorted <- posi_csv[order(posi_csv$row,posi_csv$col),]
  spot_bdy <- create_spot_id_df(posi_sorted)
  bdy_info <- data.frame(barcode=rownames(loc_rds@meta.data),bdy=loc_rds@meta.data$Location)

  for(i in 1:nrow(bdy_info)){
    spot_bdy[spot_bdy==bdy_info$barcode[i]] <- as.character(bdy_info$bdy[i])
  }

  #mask
  GeneralTypeList <- c("Mal","Bdy","nMal")
  B_mask_Mal <- create_mask(GeneralTypeList[1],spot_bdy)
  B_mask_Bdy <- create_mask(GeneralTypeList[2],spot_bdy)
  B_mask_nMal <- create_mask(GeneralTypeList[3],spot_bdy)
  B_final_mask <- array(c(B_mask_Mal,B_mask_Bdy,B_mask_nMal),dim = c(80,64,3))
  saveRDS(B_final_mask,file = paste0(rds_mask_path,sample_id,'.rds.gz'),compress = T)
  
  #cnt_csv
  gene_count <- loc_rds@assays$Spatial@counts[genelist,]
  gene_count <- as.data.frame(t(as.data.frame(gene_count)))
  rownames(gene_count) <- Map(paste0, as.character(posi_csv$col),'x', as.character(posi_csv$row)) %>% unlist
  write.csv(gene_count,paste0(csv_cnt_path,sample_id,'.csv'))
  
  #count_geneMatrix
  gene_count <- loc_rds@assays$Spatial@counts[genelist,]
  gene_count <- as.data.frame(t(as.data.frame(gene_count)))
  rownames(gene_count) <- paste0(posi_csv$row,'-',posi_csv$col)
  
  geneMatrix <- create_geneMatrix(gene_count)
  saveRDS(geneMatrix,file = paste0(rds_count_path,sample_id,'.rds.gz'),compress = T)
  
  #normed_geneMatrix
  loc_rds <- NormalizeData(loc_rds,assay = 'Spatial')
  
  gene_count <- loc_rds@assays$Spatial@data[genelist,]
  gene_count <- as.data.frame(t(as.data.frame(gene_count)))
  rownames(gene_count) <- paste0(posi_csv$row,'-',posi_csv$col)
  
  geneMatrix <- create_geneMatrix(gene_count)
  saveRDS(geneMatrix,file = paste0(rds_normedgene_path,sample_id,'.rds.gz'),compress = T)
  
  message(paste0(which(sample_list == sample_id),'/',length(sample_list),' Completed'))
}