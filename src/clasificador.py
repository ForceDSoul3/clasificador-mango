import tensorflow as tf

# Load the already trained model
modelo = tf.keras.models.load_model('model')
# Show the info about the model
modelo.summary()



# import the opencv library
import cv2
import numpy as np


def predecir(input):
    #Normalize the image
    img = np.array(input).astype(float)/255
    img = cv2.resize(img, (224,224))

    output = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(output[0])


# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # Execute analisis and print the 
    analisis = predecir(frame)
    texto = 'esperando'
    if(analisis == 1):
        texto = 'Exportacion'
    elif(analisis == 2):
        texto = 'Rechazar'
    else:
        texto = 'Aceptable'
    
    font = cv2.FONT_HERSHEY_SIMPLEX    
    cv2.putText(frame, texto, (7,70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



