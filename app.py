from flask import Flask, request, render_template, redirect, url_for
# import os
# import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
# import pickle

app = Flask(__name__)

# token_path = os.path.join(app.root_path, 'model', 'tokenizer_C4.pkl')
# model_path = os.path.join(app.root_path, 'model', 'hoax_detection_C4.tflite')
# with open(token_path, 'rb') as f:
#     tokenizer = pickle.load(f)

def predict(judulx, kontenx):
    return 0
#     interpreter = tf.lite.Interpreter(model_path)
#     interpreter.allocate_tensors()
#     news_text = [judulx, kontenx]

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
    # new_sequences = tokenizer.texts_to_sequences(news_text)
    # max_len = 100  # Make sure the maximum length matches the one used when training the model
    # new_padded = pad_sequences(new_sequences, maxlen=max_len)
    # # new_padded = preprocessing.sequence.pad_sequences(new_sequences, maxlen=max_len)

    # # Convert input data to float32 type
    # new_padded = new_padded.astype('float32')

    # # Set the input tensor with compacted data
    # interpreter.set_tensor(input_details[0]['index'], new_padded)

    # # Run the interpreter to make predictions
    # interpreter.invoke()

    # # Get the prediction result from the output tensor
    # predictions_tflite = interpreter.get_tensor(output_details[0]['index'])
    # formated_predic = f'{predictions_tflite[0]:.4f}'
    # return formated_predic

def tambah(a,b):
    return a+b

@app.route('/')
def home():
    return redirect(url_for('hoaks'))

@app.route('/hoaks', methods=['GET'])
def hoaks():
    return render_template('deteksiHoaks.html')

@app.route('/hoaks/predict', methods=['POST','GET'])
def make_prediction():
    # judul = float(request.form['judul'])
    # konten = float(request.form['konten'])
    # prediction = predict(judul, konten)
    prediction = f'{tambah(0.1, 0.35):.4f}'

    return render_template('deteksiHoaks.html', prediction=prediction)

@app.route('/bias')
def bias():
    return render_template('deteksiBias.html')

# @app.route('/hoaks', methods=['POST','GET'])
# def get_value():
#     if request.method == 'POST':
#         judul = request.form['judul']
#         konten = request.form['konten']
#         return render_template('deteksiHoaks.html', judul=judul, konten=konten)


if __name__ == '__main__':
    app.run(debug=True)
