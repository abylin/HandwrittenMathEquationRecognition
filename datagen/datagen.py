from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=False,
    fill_mode='nearest')


def gen(generator, image_path, count, save_to, prefix):
    img = load_img(image_path)  # PIL image
    x = img_to_array(img)  # Numpy array
    x = x.reshape((1,) + x.shape)
    i = count
    for data in generator.flow(x, batch_size=1, save_to_dir=save_to, save_prefix=prefix, save_format='png'):
        i -= 1
        if i < 0:
            break


gen(datagen, "right.png", 52, "gen/eval/right", "right")
gen(datagen, "left.png", 52, "gen/eval/left", "left")
gen(datagen, "right.png", 400, "gen/train/right", "right")
gen(datagen, "left.png", 400, "gen/train/left", "left")
