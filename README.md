# GPDM
Original Pytorch implementation of the GPDM model introduced in "Generating natural images with direct patch distributions matching"

![Teaser](Readme_images/Teaser_Figure.png)

# Reshuffling
`
python3 scripts/reshuffle_datasets.py
`
I added the Places50 and SIGD16 datasets from Drop-The-Gan and SinGAN so that results can be reproduced

# Retargeting
`
python3 scripts/retarget_images.py
`
Apart from the datasets from the paper I collected some interesting retargeting images in the images folder

[comment]: <> (<img src="Readme_images/Retargeting.png" width="200"/> <img src="Readme_images/Retargeting_H.png" width="200"/> )

[comment]: <> (<p align="right">)
[comment]: <> (  <img src="Readme_images/Retargeting.png" width="100" />)
[comment]: <> (  <img src="Readme_images/Retargeting_H.png" width="100" /> )
[comment]: <> (</p>)


![Teaser](Readme_images/Retargeting.png) ![Teaser](Readme_images/Retargeting_H.png)

# Retargeting
`
python3 scripts/style_transfer.py
`
In the images folder you can find images I collected from various repos and papers cited in my paper.
![Teaser](Readme_images/Style_Transfer.png)

