import streamlit as st
from streamlit.proto.Button_pb2 import Button
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
import datetime
import matplotlib.image as mpimg
import pandas as pd
import plotly.express as px

# Initital params:
IMG_SIZE = (224,224)
class_names =['1000','10000','100000','2000','20000','200000','5000','50000','500000']
img_webcam = image.load_img('/home/tamtran/Desktop/wk8_VN_currency_classification/captured.png')

########################################################################################################
                                    # PREDICTION FUNCTIONS #

def get_top_prediction(prediction, top, class_names):
    for i in range(top):
        top_confidence = list(np.sort(prediction)[0][::-1])[:top]
        top_id = list(np.argsort(prediction, axis=1)[0][::-1])[:top]
        top_class_names = [class_names[id] for id in top_id]

    return top_confidence, top_id, top_class_names

def predict(model, image_path, img_size, class_names):
    # img = image.load_img(image_path, target_size=(img_size[0], img_size[1])) #<==

    img = tf.image.resize(image_path, IMG_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    prediction = model.predict(img)
    
    top3confidence, top3id, top3name = get_top_prediction(prediction = prediction, 
                                                            top = 3, 
                                                            class_names = class_names)

    return top3confidence, top3id, top3name 

# Load model:
final_model = tf.keras.models.load_model('/home/tamtran/Desktop/wk8_VN_currency_classification/mobilenetv2_v2.h5')

final_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

########################################################################################################
                                    # LESSON LEARN FUNCTIONS #

# image augmentation:
img_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3, fill_mode='constant'),
            ])

# Preprocessing function before import image to AUGMENTATION MACHINE
def preprocessing_img(img):
    img_array = tf.image.resize(img, IMG_SIZE)
    # img_array = tf.keras.preprocessing.image.load_img(img, target_size=(400,400))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Create augmentated images function
def create_augmentated_images(img, save_dir, true_label, quantity_aug_image):
    '''
    return (`aug_directory`, `class_directory`)
    '''
    import os
    from keras.preprocessing.image import save_img
    import random

    aug_dir = save_dir+'/aug_imgs'
    class_dir = os.path.join(aug_dir, true_label)

    # make folder for augmentated images and their classes:
    try:
        os.mkdir(aug_dir)
    except:
        try:
            for cl in class_names:
                cl_path = os.path.join(aug_dir, cl)
                os.mkdir(cl_path)
        except:
            pass
    try:
        for cl in class_names:
            cl_path = os.path.join(aug_dir, cl)
            os.mkdir(cl_path)
    except:
        pass

    # Augmantating:
    total_imgs = quantity_aug_image
    for i in range(total_imgs):
        rd_num = str(int(round(random.random()*10000,0)))
        img_ar = preprocessing_img(img)
        augmented_img = img_augmentation(img_ar)
        file_path = class_dir+'/'+rd_num+true_label+'.jpg'
        save_img(file_path, augmented_img[0])
    print(f'{total_imgs} augmentated images were created in {class_dir}')

    return aug_dir, class_dir

# Make dataset from Augmentation folder:
def make_aug_ds(aug_dir, img_size, batch_size):
    import glob, os
    aug_ds = tf.keras.preprocessing.image_dataset_from_directory(
        aug_dir,
        batch_size = batch_size,
        image_size = img_size,
    )
    return aug_ds

# After fitting model, we need to save updated model and also clear out class folder for next update:
def after_fitting_model(model, class_dir):
    import os
    # Save model:
    model.save('/home/tamtran/Desktop/wk8_VN_currency_classification/mobilenetv2_v2.h5')
    print('=> Model was saved! <=')

    # Delete all files in class_dir
    files = os.listdir(class_dir)
    for f in files:
        f_path = os.path.join(class_dir, f)
        os.remove(f_path)
    print('=> All updated files been removed <=')


##########################################################################################################
##########################################################################################################
                                            # STREAMLIT #

st.title('VIETNAM DONG CLASSIFIER')
st.text('make by TRAN THANH TAM')

col1, col2 = st.beta_columns(2)

# UPLOAD IMAGE
#--------------------------------------------------------------------------------------------
with col1:
    st.header('Upload image')
    image_upload = st.file_uploader('Upload your image here:', ['jpg', 'png', 'jpeg'])

# show upload img:
if image_upload is not None:
    # preprocessing image
    img_upload_cv = np.asarray(bytearray(image_upload.read()), dtype=np.uint8)
    img_upload_cv = cv2.imdecode(img_upload_cv, 1)
    # convert GBR -> RGB
    img_upload_cv = cv2.cvtColor(img_upload_cv, cv2.COLOR_BGR2RGB)

    # show image
    st.image(img_upload_cv, channels='RGB')


# CAPTURE IMAGE
#--------------------------------------------------------------------------------------------
with col2:
    st.header('Webcam')

    show = st.checkbox('Show!')
    st.text('Press: "Space" to capture\nPress: "Esc" to exit')

    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0) # device 1/2

    while show:
        ret, frame = cam.read()
        frameshow = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frameshow)
        cv2.imshow('imshow', frame)
        k=cv2.waitKey(1)
        if k%256 == 27:
            print('escape hit!')
            break
        if k%256 == 32:
            img_name = 'captured.png'
            cv2.imwrite(img_name, frame)
            break
    else:
        cam.release()
        cv2.destroyAllWindows()
    cam.release()
    cv2.destroyAllWindows()


##########################################################################################################
                                        # PREDICT BUTTION #
col11, col22 = st.beta_columns((1,2))

with col11:
    st.text(' ')
    predict_button = st.button('Predict')
    source_prd = st.selectbox('Predict on:',['Uploaded image', 'Webcam'])
    ll_expander = st.beta_expander('LESSON LEARNT')

##########################################################################################################
                                        # Lesson Learn #

    with ll_expander:
        # st.text('hello')
        true_label_sb = st.selectbox('Tell me its true name:', ['None']+class_names)
        aug_imgs = st.slider('Number of aumentated pictures:', 5,30,10,2)
        learn_bt = st.button("Let's learn it!")
        
        if learn_bt:
            if source_prd == 'Uploaded image':
                #we have:
                save_dir = '/home/tamtran/Desktop/wk8_VN_currency_classification/'
                img_path = img_upload_cv
                true_label = true_label_sb
                BATCH_SIZE = 32

                # Make augmentated images:
                aug_dir, class_dir = create_augmentated_images(img_path, 
                                                                save_dir, 
                                                                true_label,
                                                                aug_imgs)
                st.text(f'OK! {aug_imgs} augmentated images have been created in {class_dir}')

                # Make dataset from augmentation folder:
                aug_ds = make_aug_ds(aug_dir, IMG_SIZE, BATCH_SIZE)
                st.text(f'OK! Dataset have been created!')

                # Fit with model with new data 
                history = final_model.fit(aug_ds, epochs=3)
                st.text(f'OK! Model have learnt new money!')

                # Save update model and delete files
                after_fitting_model(final_model, class_dir)
                st.text(f'OK! Model updated!')
                st.text(f'OK! removed files successfully!')
            
            elif source_prd == 'Webcam':
                #we have:
                save_dir = '/home/tamtran/Desktop/wk8_VN_currency_classification/'
                img_path = img_webcam
                true_label = true_label_sb
                BATCH_SIZE = 32

                # Make augmentated images:
                aug_dir, class_dir = create_augmentated_images(img_path, 
                                                                save_dir, 
                                                                true_label,
                                                                10)
                st.text(f'OK! 10 augmentated images have been created in {class_dir}')

                # Make dataset from augmentation folder:
                aug_ds = make_aug_ds(aug_dir, IMG_SIZE, BATCH_SIZE)
                st.text(f'OK! Dataset have been created!')

                # Fit with model with new data 
                history = final_model.fit(aug_ds, epochs=3)
                st.text(f'OK! Model have learnt new money!')

                # Save update model and delete files
                after_fitting_model(final_model, class_dir)
                st.text(f'OK! Model updated!')
                st.text(f'OK! removed files successfully!')             


##########################################################################################################
                                        # PREDICTION #
with col22:
    if source_prd == 'Webcam':
        st.image(img_webcam)

    if predict_button:
        if source_prd == 'Uploaded image':
    # img_path = '/home/tamtran/Desktop/wk8_VN_currency_classification/test_images/20k.jpeg'
            try:
                top3conf, _, top3name = predict(model=final_model, 
                                    image_path = img_upload_cv,
                                    img_size = IMG_SIZE,
                                    class_names = class_names)

                st.header(f'I think it is {top3name[0]} VND and I am sure {np.round(top3conf[0]*100, 2)}%')
                
                # Plot predict result:
                df_pred = pd.DataFrame(top3conf, index=top3name).reset_index()
                df_pred.columns = ['Money', 'Confidence']
                fig = px.bar(df_pred, x="Money", y='Confidence', color="Confidence")
                st.plotly_chart(fig, use_container_width = True)
            except:
                pass
        elif source_prd == 'Webcam':
            # img_webcam = image.load_img('/home/tamtran/Desktop/wk8_VN_currency_classification/captured.png') #<==
            # st.image(img_webcam)

            try:
                top3conf, _, top3name = predict(model=final_model, 
                                    image_path = img_webcam,
                                    img_size = IMG_SIZE,
                                    class_names = class_names)

                st.header(f'I think it is {top3name[0]} VND and I am sure {np.round(top3conf[0]*100, 2)}%')
                
                # Plot predict result:
                df_pred = pd.DataFrame(top3conf, index=top3name).reset_index()
                df_pred.columns = ['Money', 'Confidence']
                fig = px.bar(df_pred, x="Money", y='Confidence', color="Confidence")
                st.plotly_chart(fig, use_container_width = True)
            except:
                pass

##########################################################################################################
                                        # Lesson Learn #

# with col22:
#     # lesson_learnt_bt = st.button('Lesson Learnt')
#     true_label_sb = st.selectbox('Tell me its true name:', ['None']+class_names)
#     learn_bt = st.button("Let's learn new thing!")

#     if learn_bt:
#         #we have:
#         save_dir = '/home/tamtran/Desktop/wk8_VN_currency_classification/'
#         img_path = img_upload_cv
#         true_label = true_label_sb
#         BATCH_SIZE = 32

#         # Make augmentated images:
#         aug_dir, class_dir = create_augmentated_images(img_path, 
#                                                         save_dir, 
#                                                         true_label,
#                                                         10)
#         st.text(f'OK! 10 augmentated images have been created in {class_dir}')

#         # Make dataset from augmentation folder:
#         aug_ds = make_aug_ds(aug_dir, IMG_SIZE, BATCH_SIZE)
#         st.text(f'OK! Dataset have been created!')

#         # Fit with model with new data 
#         history = final_model.fit(aug_ds, epochs=3)
#         st.text(f'OK! Model have learnt new money!')

#         # Save update model and delete files
#         after_fitting_model(final_model, class_dir)
#         st.text(f'OK! Model updated!')
#         st.text(f'OK! removed files successfully!')
