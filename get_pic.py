import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image, display

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load your pretrained model (replace with your model's path)
model = tf.keras.models.load_model('asl_model.keras')

# Load the label encoder (this should be the encoder used during training)
lb = LabelEncoder()
lb.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Step 1: Preprocess the image
def preprocess_single_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Step 2: Extract landmarks using MediaPipe
def extract_single_landmark(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).reshape(1, -1)  # Reshape to (1, 63) for prediction
    else:
        return np.zeros((1, 63))  # Default to zeros if no hand is detected

# Step 3: Predict the label from the preprocessed image
def predict_from_image(image):
    preprocessed_landmarks = extract_single_landmark(image)
    
    # Ensure preprocessed_landmarks is not all zeros (i.e., no hand detected)
    if np.all(preprocessed_landmarks == 0):
        print("No hand detected in the image.")
    else:
        prediction = model.predict(preprocessed_landmarks)

        # Get the predicted label
        predicted_class = np.argmax(prediction)
        predicted_label = lb.inverse_transform([predicted_class])
        print(f"Predicted label: {predicted_label[0]}")

        print("Number of classes:", len(lb.classes_))
        print("Classes:", lb.classes_)

        # Optional: Display top 3 predictions with confidence
        top_predictions = prediction[0].argsort()[-3:][::-1]
        for idx in top_predictions:
            label = lb.classes_[idx]
            confidence = prediction[0][idx] * 100
            print(f"{label}: {confidence:.2f}% confidence")

# Step 4: Capture an image from the webcam and predict
def capture_and_predict():
    # Open the webcam (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    print("Press 'q' to capture the image and make a prediction.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the frame in a window
        cv2.imshow("Webcam Feed", frame)
        
        # Wait for the 'q' key to be pressed to capture the image and make a prediction
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # When 'q' is pressed, process the captured frame
            print("Capturing image...")
            predict_from_image(frame)
            break
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Now call the function to start capturing and predicting
capture_and_predict()
