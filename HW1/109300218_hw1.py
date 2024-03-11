import cv2
import numpy

def color_to_gray(img):
    img= 0.299*img[:,:,2] + 0.587*img[:,:,1] + 0.114*img[:,:,0]
    return img

def gray_to_binary(img):
    histogram,_= numpy.histogram(img.flatten(),256,[0,256])
    histogram= histogram.astype(float)/numpy.sum(histogram)
    optimal_threshold=0
    max_variance=0
    cumulative_sum=numpy.cumsum(histogram)
    cumulative_mean=numpy.cumsum(histogram * numpy.arange(0,256))
    for i in range(1,256):
        omega1= cumulative_sum[i]
        omega2= 1.0 - omega1
        if omega1 == 0 or omega2 == 0:
            continue
        mu1 = cumulative_mean[i] / omega1
        mu2 = (cumulative_mean[-1] - cumulative_mean[i]) / omega2
        variance = omega1 * omega2 * (mu1 - mu2) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = i
    img= numpy.where(img >= optimal_threshold, 255, 0).astype(numpy.uint8)
    return img

def downscale_no_inter(img,scale):
    height=img.shape[0]//scale
    width=img.shape[1]//scale
    temp_img=numpy.zeros((height,width,3),dtype=numpy.uint8)
    if (scale>1):
        for i in range(height):
            for j in range(width):
                temp_img[i,j]=img[i*scale,j*scale]
    return temp_img

def upscale_no_inter(img,scale):
    height=img.shape[0]*scale
    width=img.shape[1]*scale
    temp_img=numpy.zeros((height,width,3),dtype=numpy.uint8)
    if (scale>1):
        for i in range(0,height,scale):
            for j in range(0,width,scale):
                temp_img[i:i+scale,j:j+scale]=img[i//scale,j//scale]
    return temp_img

def downscale_inter(img,scale):
    height=img.shape[0]//scale
    width=img.shape[1]//scale
    half_img=numpy.zeros((height,width,3),dtype=numpy.uint8)
    for i in range(height):
        for j in range(width):
            half_img[i,j]=img[i*2,j*2]
    return half_img

def upscale_inter(img,scale):
    height = img.shape[0] * scale
    width = img.shape[1] * scale
    temp_img = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    # Upscale rows
    for i in range(0, height, scale):
        for j in range(width):
            portion = j % scale
            temp_img[i, j] = ((scale - portion) /scale* img[i // scale, j // scale] + portion/scale * img[i // scale, min((j // scale) + 1, img.shape[1] - 1)])

    for j in range(width):
            for i in range(height):
                portion = i % scale
                temp_img[i, j] = ((scale - portion) / scale * temp_img[i - portion, j] + portion / scale * temp_img[min(i - portion + scale, height - 1), j])

    return temp_img

def resize_inter(img):
    return img

def main():
    image_paths=["images/img1.png","images/img2.png","images/img3.png"]
    for image_path in image_paths:
        image=cv2.imread(image_path,1)
        #cv2.imwrite("results/"+image_path.split('/')[1].split('.')[0]+"_q1-1.png",color_to_gray(image))
        #cv2.imwrite("results/"+image_path.split('/')[1].split('.')[0]+"_q1-2.png",gray_to_binary(color_to_gray(image)))
        #cv2.imwrite("results/"+image_path.split('/')[1].split('.')[0]+"_q2-1-half.png",downscale_no_inter(image,2))
        #cv2.imwrite("results/"+image_path.split('/')[1].split('.')[0]+"_q2-1-double.png",upscale_no_inter(image,2))
        cv2.imwrite("results/"+image_path.split('/')[1].split('.')[0]+"_q2-2-double.png",upscale_inter(image,2))
    return 0

if __name__ == "__main__":
    main()


