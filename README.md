# autoCropper

This python script auto crops images into four quadrants, of equal proportion based on their height and width dimensions (at the pixel level).

Requirements:

**PIL: from PIL import Image**

**open cv2: import cv2**

**numpy: import numpy as np**

________

## Example Input Image:

```Python
from openCV_autoCropper import quadCropper

# dictionary specifies the file, and the desired cropped outcome
file_dictionary = {'myImg.jpg': 1, 'myImg.jpg': 2, 'myImg.jpg': 3, 'myImg.jpg': 4}


quadCropper(image_dir, crop_dir, file_dictionary)

```
(This is just one image)


![](image_dir/RSIP_Example_HipSegmentation.jpg)

## Example Output Images:
Produces four cropped quadrants of roughly equal size.


![](crop_dir/RSIP_Example_HipSegmentation2_cropped.jpg). ![](crop_dir/RSIP_Example_HipSegmentation1_cropped.jpg)

![](crop_dir/RSIP_Example_HipSegmentation3_cropped.jpg). ![](crop_dir/RSIP_Example_HipSegmentation4_cropped.jpg)



