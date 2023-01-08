import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf, matplotlib.pyplot as plt, pandas as pd
from model import model, checkpoint_path

def preprocess_data(df):
	filenames = df.pop('filename')
	age = tf.keras.Input(shape=1, name="age", dtype=tf.float32)
	race = tf.keras.Input(shape=1, name="race", dtype=tf.float32)
	gender = tf.keras.Input(shape=1, name="gender", dtype=tf.float32)
	inputs = {
		"age": age,
		"race": race,
		"gender": gender
	}
	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(df['age'])
	vocab = [0, 1, 2, 3]
	lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode="one_hot")
	preprocessed = [
		normalizer(age),
		gender,
		lookup(race)
	]
	preprocessor = tf.keras.Model(df, )

if __name__ == "__main__":
	tf.keras.utils.plot_model(model)
	model.summary()
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
	df = pd.read_csv("data.csv")
	#preprocess_data(df)
	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		rescale = 1 / 255.0,
		rotation_range=20,
		zoom_range=0.05,
		shear_range=0.05,
		horizontal_flip=True,
		validation_split=0.2,
	)
	batch_size = 32
	base_dir = "UTKFace/"
	train_generator = train_datagen.flow_from_dataframe(
		dataframe=df,
		directory=base_dir,
		x_col="filename",
		y_col=["age"],
		target_size=(200, 200),
		batch_size=batch_size,
		class_mode="raw",
		#classes=["White", "Black", "Asian", "White", "Others"],
		color_mode="grayscale",
		subset="training",
		shuffle=True,
		seed=78,
	)

	val_generator = train_datagen.flow_from_dataframe(
		dataframe=df,
		directory=base_dir,
		x_col="filename",
		y_col=["age"],
		target_size=(200, 200),
		batch_size=batch_size,
		#classes=["White", "Black", "Asian", "White", "Others"],
		class_mode="raw",
		color_mode="grayscale",
		subset="validation",
		shuffle=True,
		seed=78
	)
	epochs = 15
	history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[cp_callback])
	gender_acc = history.history['gender_out_accuracy']
	gender_val_acc = history.history['val_gender_out_accuracy']
	age_loss = history.history['age_out_loss']
	age_val_loss = history.history['val_age_out_loss']
	race_acc = history.history["race_out_accuracy"]
	race_val_acc = history.history["val_race_out_accuracy"]
	print(gender_acc)
	print(gender_val_acc)
	print(age_loss)
	print(age_val_loss)
	print(race_acc)
	print(race_val_acc)
	plt.plot(range(epochs), gender_acc, 'r', label="Training Accuracy")
	plt.plot(range(epochs), gender_val_acc, 'b', label="Validation Accuracy")
	plt.title("Gender Accuracy")
	plt.legend()
	plt.show()

	plt.plot(range(epochs), age_loss, 'r', label="Training Loss")
	plt.plot(range(epochs), age_val_loss, 'b', label="Validation Loss")
	plt.title("Age Loss")
	plt.legend()
	plt.show()

	plt.plot(range(epochs), race_acc, 'r', label="Training Accuracy")
	plt.plot(range(epochs), race_val_acc, 'b', label="Validation Accuracy")
	plt.title("Race Accuracy")
	plt.legend()
	plt.show()
