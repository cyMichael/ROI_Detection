# Transfer learning on TCGA-SKCM

## Overview
To better evaluate the method, the model is deployed to the TCGA skin cancer dataset. The model is directly applied to the TCGA dataset without fine-tuning since patch-level labels are not available for TCGA dataset. For some TCGA slides, there are some annotations on them, but we are not sure what the annotations are made for. Even though the color, scan method and many other factors make the TCGA dataset very different from the melanocytic dataset used in the paper, the model is able to find out annotated regions in those TCGA slides. See some examples below. More results can be found at folder [images](/tcga_visual/images).

## Examples
<img src="./images/TCGA-D3-A2JD-06Z-00-DX1_B6DBA83D-6C77-4F73-87B8-30487C8AB7C1.png" width="400"> <img src="./images/TCGA-D3-A2JD-06Z-00-DX1_B6DBA83D-6C77-4F73-87B8-30487C8AB7C1_heat_.png" width="400">

<img src="./images/TCGA-D3-A51J-06Z-00-DX1_A292E5D2-B00F-400B-875D-6C57E215A29E.png" width="400"> <img src="./images/TCGA-D3-A51J-06Z-00-DX1_A292E5D2-B00F-400B-875D-6C57E215A29E_heat_.png" width="400">

<img src="./images/TCGA-EB-A82C-01Z-00-DX1_11DBF5EC-C481-4F3D-A6EA-88644D7487A7.png" width="400"> <img src="./images/TCGA-EB-A82C-01Z-00-DX1_11DBF5EC-C481-4F3D-A6EA-88644D7487A7_heat_.png" width="400">

<img src="./images/TCGA-EB-A6QY-01Z-00-DX1_F6C8B1E9-42E1-4F97-AADF-6A6C14637BBF.png" width="400"> <img src="./images/TCGA-EB-A6QY-01Z-00-DX1_F6C8B1E9-42E1-4F97-AADF-6A6C14637BBF_heat_.png" width="400">

<img src="./images/TCGA-D3-A8GC-06Z-00-DX1_F46C95E6-FF24-431F-9E86-1C29CE71524D.png" width="400"> <img src="./images/TCGA-D3-A8GC-06Z-00-DX1_F46C95E6-FF24-431F-9E86-1C29CE71524D_heat_.png" width="400">

## Results on Other Example

<img src="./images/TCGA-D3-A3C1-06Z-00-DX1_F6C5DABC-9FEB-4A55-A764-6958046CFE39.png" width="400"> <img src="./images/TCGA-D3-A3C1-06Z-00-DX1_F6C5DABC-9FEB-4A55-A764-6958046CFE39_heat_.png" width="400">
