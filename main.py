import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image,ImageChops,ImageOps
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def FindForm(imageName):
    image = cv2.imread(imageName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y + h, x:x + w]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite('cropped_field.png', cropped_image)
    return cropped_image

def sliceImage(image):
    height, width, _ = image.shape
    fragment_width = width // 9
    remainder = width % 9

    cells = []
    start_x = 2

    for i in range(9):
        end_x = start_x + fragment_width
        if i < remainder:
            end_x += 1
        print(f"Fragment {i}: start_x={start_x}, end_x={end_x}, width={end_x - start_x}")  # Print for debugging
        fragment = image[:, start_x:end_x]
        cells.append(fragment)
        start_x = end_x

    return cells

def preprocess_cells(cells):
    processed_cells = []
    i = 0
    for cell in cells:
        gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        height, width = gray_cell.shape
        top_crop = int(0.29 * height)
        bottom_crop = int(0.29 * height)
        cropped_cell = gray_cell[top_crop:height - bottom_crop, :]

        height = cropped_cell.shape[0]
        if cropped_cell.shape[1] > height:
            excess = cropped_cell.shape[1] - height
            left_margin = excess // 2
            right_margin = excess - left_margin
            square_cell = cropped_cell[:, left_margin:cropped_cell.shape[1] - right_margin]
        else:
            square_cell = cropped_cell

        inverted_cell = cv2.bitwise_not(square_cell)
        resized_cell = cv2.resize(inverted_cell, (28, 28))
        normalized_cell = resized_cell / 255.0

        _, binary_cell = cv2.threshold(normalized_cell, 0.5, 1.0, cv2.THRESH_BINARY)
        normalized_cell = np.expand_dims(binary_cell, axis=-1)
        processed_cells.append(normalized_cell)
        i = i + 1

    return np.array(processed_cells)

##################### IMAGE CROP

def normalize_img(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label

def augment_img(image, label):

    translation = tf.random.uniform([], -10, 10, dtype=tf.int32)
    image_width = tf.shape(image)[1]
    target_width = 28
    max_translation = image_width - target_width
    translation = tf.clip_by_value(translation, 0, max_translation)
    image = tf.image.resize_with_crop_or_pad(image, target_height=28, target_width=28)
    image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=translation,
                                          target_height=28, target_width=28)
    image = tf.image.resize(image, (28, 28))
    image = tf.image.random_crop(image, size=[28, 28, 1])
    return image, label

def CNN():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ''' 
    for image, label in ds_train.take(1):  # 'take(1)' bierze tylko jeden obrazek
        # Przekształć obrazek do formatu NumPy
        image = image.numpy()

        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label.numpy()}')  
        plt.axis('off')  
        plt.show()
    '''

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Model CNN
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=9,
        validation_data=ds_test,
    )

    tf.keras.models.save_model(model, "cnn.keras")
    print("Model saved")
    print(f"CNN: {model.evaluate(ds_test)}")

    return model

###################MODEL


def main():
    form = FindForm('form2.png')
    cells = sliceImage(form)
    preprocessed_cells = preprocess_cells(cells)
    model = CNN()
    #model = tf.keras.models.load_model("cnn.keras")
    predictions = model.predict(preprocessed_cells)
    for i, prediction in enumerate(predictions):
        predicted_label = np.argmax(prediction)
        print(f"Fragment {i}: Predicted label = {predicted_label}")


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
