import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs"; // TensorFlow.js for browser
import sharp from "sharp";

let model;

const loadModel = async () => {
  if (!model) {
    model = await mobilenet.load({
      version: 1,
      alpha: 0.25, // Smallest model variant
    });
  }
  return model;
};

const decodeImage = (base64Image) => {
  return new Promise((resolve, reject) => {
    const buffer = Buffer.from(base64Image, "base64");
    sharp(buffer)
      .resize(427, 640)
      .raw()
      .toBuffer()
      .then((data) => {
        const width = 427;
        const height = 640;
        const channels = 3;
        if (data.length !== width * height * channels) {
          reject(
            new Error("Decoded image data does not match the expected size.")
          );
          return;
        }
        const tensor = tf.tensor(new Uint8Array(data), [
          height,
          width,
          channels,
        ]);
        resolve(tensor);
      })
      .catch(reject);
  });
};

export async function POST(request) {
  try {
    const { imageData } = await request.json();

    const imageTensor = await decodeImage(imageData);
    const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);

    const model = await loadModel();
    const predictions = await model.classify(resizedImageTensor);

    imageTensor.dispose();
    resizedImageTensor.dispose();

    return new Response(JSON.stringify({ predictions }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error) {
    console.error("Error:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  }
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
