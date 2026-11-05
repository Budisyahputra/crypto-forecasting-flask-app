import os
import zipfile
import tempfile
import tensorflow as tf

def build_attention_lstm(lookback=60):
    inp = tf.keras.layers.Input(shape=(lookback, 1), name="input_layer")

    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm")(inp)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)

    att = tf.keras.layers.Dense(1, activation="linear", name="dense_att")(x)
    att = tf.keras.layers.Softmax(axis=1, name="softmax_att")(att)

    xw = tf.keras.layers.Multiply(name="multiply_att")([x, att])

    xsum = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1),
        output_shape=(64,),
        name="reduce_sum_att"
    )(xw)

    x2 = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(xsum)
    out = tf.keras.layers.Dense(1, activation="linear", name="out")(x2)

    return tf.keras.Model(inp, out, name="attention_lstm_fixed")

def extract_weights_h5(keras_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(keras_path, "r") as z:
        names = z.namelist()
        candidates = [n for n in names if n.endswith(".h5") and "weights" in n.lower()]
        if not candidates:
            candidates = [n for n in names if n.endswith(".h5")]
        if not candidates:
            raise RuntimeError(f"Tidak menemukan file .h5 di dalam {keras_path}. Isi zip: {names[:30]}")

        weights_name = candidates[0]
        out_path = os.path.join(out_dir, os.path.basename(weights_name))
        with z.open(weights_name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        return out_path

def convert_one(in_keras: str, out_dir: str, lookback: int = 60):
    print(f"[LOAD] {in_keras}")

    model = build_attention_lstm(lookback=lookback)

    with tempfile.TemporaryDirectory() as tmp:
        wpath = extract_weights_h5(in_keras, tmp)
        print(f"[WEIGHTS] {wpath}")
        model.load_weights(wpath)

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    # Keras 3: export untuk SavedModel folder
    model.export(out_dir)
    print(f"[EXPORTED] {out_dir}\n")

if __name__ == "__main__":
    LOOKBACK = 60
    convert_one("models/BTC-USD_H1_L60_close.keras", "models_fixed/BTC_savedmodel", lookback=LOOKBACK)
    convert_one("models/ETH-USD_H1_L60_close.keras", "models_fixed/ETH_savedmodel", lookback=LOOKBACK)
    convert_one("models/BNB-USD_H1_L60_close.keras", "models_fixed/BNB_savedmodel", lookback=LOOKBACK)
