# GPDM

Original Pytorch implementation of the GPDM algorithm introduced in

["Generating Natural Images with Direct Patch Distribution Matching"](https://arxiv.org/abs/2203.11862)

Accepted to [ECCV 2022](https://eccv2022.ecva.net/)

## [**Live-demo**](https://replicate.com/ariel415el/gpdm) | [**Paper**](https://arxiv.org/abs/2203.11862)

![Teaser](Readme_images/Teaser_Figure.jpg)

## Video presentation

<div align="center">
  <a href="https://www.youtube.com/watch?v=_7gKxR6MVQU"><img src="https://img.youtube.com/vi/_7gKxR6MVQU/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

# Run GPDM:

### Reshuffling

`$ python3 main.py data/images/SIGD16/7.jpg`

| Input                                            | Output                                               |
|--------------------------------------------------|------------------------------------------------------| 
| <img src=data/images/SIGD16/7.jpg height="150"/> | <img src="Readme_images/reshuffle.png" height="150"/>|  

### Retargeting

`$ python3 main.py data/images/SIGD16/4.jpg --init_from target --width_factor 1.5`

| Input                                            | Output                                               |
|--------------------------------------------------|------------------------------------------------------| 
| <img src=data/images/SIGD16/4.jpg height="150"/> | <img src="Readme_images/retarget.png" height="150"/> |  

### Style transfer

`$ python3 main.py data/images/style_transfer/style/mondrian.jpg --init_from data/images/style_transfer/content/trump.jpg
--fine_dim 1024 --coarse_dim 256 --noise_sigma 0`

| Input                                                                 | init_from                                                 | Output                                                     |
|-----------------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------| 
| <img src="data/images/style_transfer/style/mondrian.jpg" height="200"/> | <img src=data/images/style_transfer/content/trump.jpg height="200"/> | <img src="Readme_images/style_transfer.png" height="200"/> |  

### Texture synthesis

`$ python3 main.py data/images/textures/cobbles.jpeg --width_factor 1.5 --height_factor 1.5`

| Input                                                     | Output                                                            |
|-----------------------------------------------------------|-------------------------------------------------------------------| 
| <img src=data/images/textures/cobbles.jpeg height="100"/> | <img src="Readme_images/texture_synthesis.png" height="150"/> |  

# Reproduce paper tables

I added the Places50 and SIGD16 datasets from [Drop-The-Gan](https://www.wisdom.weizmann.ac.il/~vision/gpnn/)
and [SinGAN](https://tamarott.github.io/SinGAN.htm) so that results can be reproduced

Apart from the datasets from the paper I collected
some interesting retargeting images in the images folder

In the images folder you can find images I collected from various repos and papers cited in my paper.

# Cite

```
@inproceedings{elnekave2022generating,
  title={Generating natural images with direct Patch Distributions Matching},
  author={Elnekave, Ariel and Weiss, Yair},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XVII},
  pages={544--560},
  year={2022},
  organization={Springer}
}
```
