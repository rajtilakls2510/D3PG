from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model

inp = layers.Input(shape=(2,))
x = layers.Dense(5)(inp)
out = layers.Dense(1)(x)
model = Model(inputs=inp, outputs=out)
print(type(model))
print(model.summary())

model.save("some_model")

model2 = load_model("lunar_lander_cont_agent/actor_network")
print(type(model2))
print(model2.get_weights())