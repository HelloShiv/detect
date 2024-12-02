const mobilenet = require('@tensorflow-models/mobilenet');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

(async () => {
  try {
    // Path to your local image file
    const imagePath = 'src/app/api/detect/cat.jpg';

    // Read the image file into a buffer
    const imageBuffer = fs.readFileSync(imagePath);

    // Decode the image buffer into a Tensor
    const imageTensor = tf.node.decodeImage(imageBuffer);

    // Load the MobileNet model
    const model = await mobilenet.load();

    // Classify the image
    const predictions = await model.classify(imageTensor);

    // Output the predictions
    console.log('Predictions:', predictions);

    // Dispose of the tensor to free memory
    imageTensor.dispose();
  } catch (error) {
    console.error('Error:', error);
  }
})();
