import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs-node';  // TensorFlow.js Node.js package
import fs from 'fs';
import path from 'path';

// Force the backend to be 'tensorflow' (Node.js backend)
tf.setBackend('tensorflow').then(() => {
  console.log('TensorFlow.js backend set to "tensorflow".');
}).catch((error) => {
  console.error('Failed to set TensorFlow.js backend:', error);
});

// Cache the MobileNet model to load it only once per deployment
let model;
const loadModel = async () => {
  if (!model) {
    // Load the smallest MobileNet model (version 1, alpha 0.25)
    model = await mobilenet.load({
      version: 1,
      alpha: 0.25, // Smallest model variant
    });
  }
  return model;
};

export async function POST(req) {
  try {
    // Parse the request to get the image file path
    const { imagePath } = await req.json();

    // Ensure the path is valid and resolve it relative to the server
    const resolvedPath = path.resolve(process.cwd(), imagePath);

    if (!fs.existsSync(resolvedPath)) {
      return new Response(JSON.stringify({ error: 'File not found' }), { status: 404 });
    }

    // Read the image file into a buffer
    const imageBuffer = fs.readFileSync(resolvedPath);

    // Decode and resize the image buffer into a Tensor to optimize processing time
    let imageTensor = tf.node.decodeImage(imageBuffer);
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);  // Resize to MobileNet input size (224x224)

    // Load the model (use cached model if already loaded)
    const model = await loadModel();

    // Classify the image
    const predictions = await model.classify(imageTensor);

    // Dispose of the tensor to free memory
    imageTensor.dispose();

    // Return the predictions as JSON
    return new Response(JSON.stringify({ predictions }), { status: 200 });
  } catch (error) {
    console.error('Error:', error);
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }
}
