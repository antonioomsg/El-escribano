def resize_image(imagen,size,output=False):
    im = Image.open(imagen)
    rgb_image = RGBA_a_RGB(im)
    im_rgb = rgb_image.resize(size)
    if not output:
        return im_rgb 
    onlyfiles = [f for f in listdir(output) if isfile(join(output, f))]
    if image not in onlyfiles:
        im.save(output)
        return im_rgb
    else:
        print("file already exists, continuing to next one") 
        return im_rgb

def RGBA_a_RGB(image1):
    if image1.mode == "RGBA":
        background = Image.new("RGB", image1.size, (255, 255, 255))
        background.paste(image1, mask = image1.split()[3])
        #background.save(f"{image1}", "JPEG", quality=100)
        #rgb_image = Image.open(f"{image1}")
        return background
    return image1

def image_to_array(image):
    im_np= np.asarray(image)
    one_line=im_np.ravel()
    return one_line