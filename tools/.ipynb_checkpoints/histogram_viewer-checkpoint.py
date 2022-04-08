from PIL import Image
from pylab import *
import sys


if __name__ == "__main__":
    img_file1 = sys.argv[1]
    # img_file2 = sys.argv[2]
    
    img1 = array(Image.open(img_file1))
    # img2 = array(Image.open(img_file2))
    figure()

    hist(img1.flatten(),256,color='red')
    # hist(img2.flatten(),256,color='blue')
    
    show()


