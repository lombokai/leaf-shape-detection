import cv2, numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image

def set_gpu_mem_alloc(mem_use):
    avail  = 2004
    percent = mem_use / avail
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = percent
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

set_gpu_mem_alloc(1500)

CLASSES = ["Melengkung", "Menjari", "Menyirip", "Sejajar"]

video_capture = cv2.VideoCapture(0) # Set to 1 for front camera
video_capture.set(4, 224) # Width
video_capture.set(5, 224) # Height

from keras.models import load_model, model_from_json
# model = load_model('saved/leaf_cnn.h5')

json_file = open('saved/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved/model.h5")
print("Loaded model from disk")

# Start the video capture loop
while (True):
    ret, original_frame = video_capture.read()
    # Resize the frame to fit the imageNet default input size
    frame_to_predict = cv2.resize(original_frame, (224, 224))

    resized_image = np.expand_dims(frame_to_predict, axis=0)

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mse'])
    # score = loaded_model.evaluate(frame_to_predict, CLASSES, verbose=0)

    prediction = loaded_model.predict(resized_image)

    hasil = CLASSES[np.argmax(prediction)]

    score = loaded_model.evaluate(resized_image, prediction, verbose=0)

    cv2.putText(original_frame, "Label: %s " % (hasil),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    # Display the video
    cv2.imshow("Classification", original_frame)

    # Hit q or esc key to exit
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break