import smtplib
recipients = [
    "iamnevdread@gmail.com",
    "anshika.choudhary0502@gmail.com",
             ]
directorys = ["models", "chkpt"]
# variables
checkpoint_path = "chkpt"
save_path = "models"
batch_size = 8
epochs = 40

def createDirectories(directorys):
    # Directory path
    
    for directory in directorys:
        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory if it doesn't exist
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")


def send_email(subject, body, sender, recipients, password):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(sender, recipients, message)
        print("Email sent successfully!")
    except Exception as e:
        print("An error occurred while sending the email:", str(e))
    finally:
        server.quit()

def sendState(message):
# Example usage
    subject = "PHAS AUTO GENERATED MAIL FOR UNET MODEL STATUS"
    body = str(message)
    sender = "devendradhayal1203@gmail.com"
    global recipients 
    password = "elsz chdo chyb qrqr"
    
    print(message)

    # send_email(subject, body, sender, recipients, password)

# import required libraries
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

##hush warnings
import shutup;
shutup.please()

# Read the CSV file
data = pd.read_csv('DL_info.csv')

# Define the input shape of the images
input_shape = (512, 512, 1)

# Preprocess the data
def preprocess_data(data, input_shape, base_dir):
    images = []
    masks = []
    read= 0 
    unread  =0 
    
    for _, row in data.iterrows():
        
        patient_index = row['Patient_index']
        study_index = row['Study_index']
        series_index = row['Series_ID']
        slice_index = row['Key_slice_index']
        bounding_boxes = eval(row['Bounding_boxes'])
       
        le = len(str(slice_index))
        p_name = row['File_name']
        file_name = p_name[len(p_name)-7:]
        # Construct the file path based on the naming convention
       
        p_name = p_name[:len(p_name)-le-6]
        file_path = os.path.join(base_dir, p_name, file_name)
        
        # Load the 16-bit grayscale image
        try :
       
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, input_shape[:2])
            read = read +1
        except:
      
            unread = unread+1
            data = data[data['Key_slice_index'] != slice_index ]
            continue
        # Apply windowing
        window_center = 40
        window_width = 80
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        image = np.clip(image, min_value, max_value)
        
        # Normalize the image to 0-1 range
        image = (image - min_value) / (max_value - min_value)
        images.append(image)
        
        # Create the mask from bounding boxes
        mask = np.zeros(input_shape[:2], dtype=np.uint8)
        bbox  = bounding_boxes 
        
        # bbox = list(map(float, bbox.split(',')))
        x1, y1, x2, y2 = int(float(bbox[0])) , int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))
       
        mask[y1:y2, x1:x2] = 1
        masks.append(mask)
    
    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)
    print("read:", read, "unread :",unread)
    return images, masks

# Set the base directory where the images are stored
base_dir = 'Images_png'
def RunPreprocess(data, input_shape , base_dir,):
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=49)

    # Preprocess the training and testing data
    train_images, train_masks = preprocess_data(train_data, input_shape, base_dir)
    test_images, test_masks = preprocess_data(test_data, input_shape, base_dir)
    sendState(f"Dataset Loaded with train size {len(train_images)} and test {len(test_images)} ")

    print("train size(images masks) :",len(train_images),len(train_masks))
    return train_images, train_masks, test_images, test_masks

train_images, train_masks, test_images, test_masks = RunPreprocess(data, input_shape , base_dir)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Unet model 



def unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder path
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder path
    up5 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# init U-Net model

model = unet(input_shape)
summary = str(model.to_json())
sendState(str(summary))

def runDevice(device):
    global train_images , train_masks, batch_size, epochs ,test_images, test_masks
    if device == 'cpu':
        with tf.device('/CPU:0'):
             
            sendState("Failed to  run on gpu Not Enough memory, trying cpu")    
            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"epoch.weights.h5",save_weights_only=True,verbose=1)
        
        
            # Train the model
            sendState(f"model started training epoch {epochs} batch size {batch_size}")
            model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_masks),callbacks=[cp_callback])
            sendState("model done training, Starting test")
            model.save_weights('./chkpt/finalChkpt.weights.h5')

            # Evaluate the model on the testing set
            loss, accuracy = model.evaluate(test_images, test_masks)
            
            sendState(f'Done with testing \n Test Loss: {loss:.4f} Test Accuracy: {accuracy:.4f}')
        
    else :
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"epoch.weights.h5",save_weights_only=True,verbose=1)
    

        # Train the model
        sendState(f"model started training epoch {epochs} batch size {batch_size}")
        model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_masks),callbacks=[cp_callback])
        sendState("model done training, Starting test")
        model.save_weights('./chkpt/finalChkpt.weights.h5')

        
        # Evaluate the model on the testing set
        loss, accuracy = model.evaluate(test_images, test_masks)
        sendState(f'Done with testing \n Test Loss: {loss:.4f} Test Accuracy: {accuracy:.4f}')


try :   
    
except :
    runDevice("CPU")
        
if __name__ == "__main__":

    # option 1: execute code with extra process
    p = multiprocessing.Process(target=runDevice("GPU"))
    p.start()
    p.join()

    # wait until user presses enter key
    raw_input()

    # option 2: just execute the function
    run_tensorflow()

    # wait until user presses enter key
    raw_input()