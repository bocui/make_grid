# make_grid
Add something from branch host
it is for the layout of a big image. 
If our batch data is the shape of [N, H, W, C]
We want to arange it in the layout that 
[x_pic * y_pic] of small images. 
Besides, around each small image, we choose to add padding. i.e. 
padding_up, padding_down, padding_left, padding_right
