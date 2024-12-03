import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs-node'; // Use tfjs-node for server-side
import sharp from 'sharp'; // Image processing library

let model;
const loadModel = async () => {
  if (!model) {
    // Load the MobileNet model (version 1, alpha 0.25)
    model = await mobilenet.load({
      version: 1,
      alpha: 0.25, // Smallest model variant
    });
  }
  return model;
};

export async function POST(request) {
  try {
    // Parse the incoming request data
    const { imageData } = await request.json();

    // Decode the base64 image
    const buffer = Buffer.from(imageData, 'base64');

    // Use Sharp to decode and resize the image
    const image = await sharp(buffer)
      .resize(224, 224)
      .toBuffer();

    // Convert the image buffer into a tensor
    const imageTensor = tf.node.decodeImage(image);

    // Load the model (use cached model if already loaded)
    const model = await loadModel();

    // Classify the image
    const predictions = await model.classify(imageTensor);

    // Dispose of tensors to free memory
    imageTensor.dispose();

    // Return predictions as JSON response
    return new Response(JSON.stringify({ predictions }), { status: 200 });
  } catch (error) {
    console.error('Error:', error);
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }
}
