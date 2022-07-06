# GPDM
Original Pytorch implementation of the GPDM algorithm introduced in

["Generating Natural Images with Direct Patch Distribution Matching"](https://arxiv.org/abs/2203.11862)

![Teaser](Readme_images/Teaser_Figure.png)

## [**Live-demo**](https://replicate.com/ariel415el/gpdm) | [**Paper**](https://arxiv.org/abs/2203.11862)


# Reshuffling
`
python3 scripts/reshuffle_datasets.py
`

I added the Places50 and SIGD16 datasets from Drop-The-Gan and SinGAN so that results can be reproduced

# Retargeting
`
python3 scripts/retarget_images.py
`

Apart from the datasets from the paper I collected 
some interesting retargeting images in the images folder

<img src="Readme_images/Retargeting.png" height="400"/> <img src="Readme_images/Retargeting_H.png" height="400"/>

<!-- ![Teaser](Readme_images/Retargeting.png) ![Teaser](Readme_images/Retargeting_H.png) -->

# Style transfer
`
python3 scripts/style_transfer.py
`

In the images folder you can find images I collected from various repos and papers cited in my paper.

![Teaser](Readme_images/Style_Transfer.png)

# Cite
@article{elnekave2022generating,
  title={Generating natural images with direct Patch Distributions Matching},
  author={Elnekave, Ariel and Weiss, Yair},
  journal={arXiv preprint arXiv:2203.11862},
  year={2022}
}