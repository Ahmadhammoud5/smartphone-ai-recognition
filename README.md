Smartphone AI Recognition & Price Prediction
This project was built as part of our final year in Computer and Communications Engineering. The idea came from a simple problem: if you upload a photo of a phone, can an AI tell you what model it is and how much it's worth?
We built a 3-stage pipeline to do exactly that.

How it works
Stage 1 — Is there a phone in the image?

A CNN classifier looks at the image and decides if it contains one phone, no phone, or multiple phones. We trained it on around 8,000 images and it reached 90% accuracy on the test set. We also trained a Random Forest on the same task as a baseline — it got 74%, which confirmed that deep learning handles images much better than traditional ML.
Stage 2 — Which iPhone model is it?

If the image contains a phone, a YOLOv8 model tries to identify the exact iPhone model across 16 classes (from iPhone 7 to iPhone 15 Pro). This was the hardest part because many iPhone generations look nearly identical from the front. We reached mAP50 of 0.637. Best results were on iPhone 14 Pro and iPhone 12 Pro. Weaker ones like iPhone 11 and iPhone 13 are just visually too similar.
Stage 3 — What is it worth?

A Random Forest takes structured inputs (model, storage, battery health, condition) and predicts the resale price. R² of 0.9936 on the test set.
The whole thing is wrapped in a Streamlit app where you upload an image and get all three outputs.

Tech used

TensorFlow / Keras for the CNN
Ultralytics YOLOv8 for object detection
scikit-learn for Random Forest and Decision Tree
Streamlit for the demo app
Python throughout


Run it locally
bashgit clone https://github.com/Ahmadhammoud5/smartphone-ai-recognition.git
cd smartphone-ai-recognition
pip install -r requirements.txt
streamlit run app/streamlit_app.py
