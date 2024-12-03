const sharp = require('sharp');
const tf = require('@tensorflow/tfjs'); // TensorFlow.js (client-side)
const fs = require('fs');

const imagePath = 'cat.jpg';
sharp(imagePath)
  .resize(427, 640) // Match the original resolution
  .raw()
  .toBuffer()
  .then((buffer) => {
    const tensor = tf.tensor(new Uint8Array(buffer), [427, 640, 3]);
    console.log(tensor);
  })
  .catch(err => console.error(err));




console.log("other one is: ")

  const tf1 = require('@tensorflow/tfjs-node');

  
  
  const buffer = fs.readFileSync(imagePath);
  const tensor = tf1.node.decodeImage(buffer);
  console.log(tensor);
  