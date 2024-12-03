import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';  // TensorFlow.js for browser

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

const sharp = require('sharp');

const decodeImage = (base64Image) => {
  return new Promise((resolve, reject) => {
    // Convert base64 to buffer
    const buffer = Buffer.from(base64Image, 'base64');
    
    // Use sharp to decode and resize the image
    sharp(buffer)
      .resize(427, 640) // Resize to the desired resolution
      .raw() // Get the raw image data
      .toBuffer()
      .then((data) => {
        // The raw image data from sharp is flat. The format will be [width, height, channels] (in this case RGB)
        const width = 427;
        const height = 640;
        const channels = 3; // RGB

        // Ensure the image data size matches the expected number of values
        if (data.length !== width * height * channels) {
          reject(new Error('Decoded image data does not match the expected size.'));
          return;
        }

        // Convert the raw image buffer into a tensor
        const tensor = tf.tensor(new Uint8Array(data), [height, width, channels]);
        resolve(tensor); // Resolve with the tensor
      })
      .catch(reject); // Reject if there's an error during processing
  });
};


export async function POST(request) {
  try {
    const { imageData } = await request.json();
    
    // Decode the base64 image to a tensor using the browser's canvas API
    const imageTensor = await decodeImage(imageData);

    // Resize the image to match MobileNet input size (224x224)
    const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);

    // Load the model (use cached model if already loaded)
    const model = await loadModel();

    // Classify the image
    const predictions = await model.classify(resizedImageTensor);

    // Dispose of tensors to free memory
    imageTensor.dispose();
    resizedImageTensor.dispose();

    // Return predictions as JSON response
    return new Response(JSON.stringify({ predictions }), { status: 200 });
  } catch (error) {
    console.error('Error:', error);
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }
}
