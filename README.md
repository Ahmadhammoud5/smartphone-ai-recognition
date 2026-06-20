Smartphone AI Recognition & Price Prediction
The idea is simple: upload a photo of a phone, and the system tells you what model it is and how much it's worth.
To do that we built a 3-stage pipeline.

How it works
Stage 1 — Is there a phone in the image?

A CNN classifier looks at the image and decides if it contains one phone, no phone, or multiple phones. Trained on around 8,000 images, it reached 90% accuracy on the test set. We also tested a Random Forest on the same task as a comparison — it got 74%, which shows how much better deep learning handles image data.
Stage 2 — Which iPhone model is it?

If a phone is detected, a YOLOv8 model identifies the exact iPhone model across 16 classes (iPhone 7 through iPhone 15 Pro). This was the hardest part because many iPhone generations look nearly identical from the front. We reached mAP50 of 0.637. Strongest results were on iPhone 14 Pro and iPhone 12 Pro. Models like iPhone 11 and iPhone 13 were harder simply because the visual differences are minimal.
Stage 3 — What is it worth?

A Random Forest takes structured inputs (model, storage, battery health, condition) and predicts the resale price. R² of 0.9936 on the test set.
Everything is wrapped in a Streamlit app where you upload an image and get all three outputs.

Tech used

TensorFlow / Keras
Ultralytics YOLOv8
scikit-learn
Streamlit
Python


Run it locally
bashgit clone https://github.com/Ahmadhammoud5/smartphone-ai-recognition.git
cd smartphone-ai-recognition
pip install -r requirements.txt
streamlit run app/streamlit_app.py
