const path = require('path');
const fs = require('fs');

// Use absolute path for the image file
const imagePath = path.resolve(__dirname, 'cat.jpg'); // __dirname ensures the current directory
console.log('Image path:', imagePath);

fs.readFile(imagePath, (err, data) => {
  if (err) {
    console.error('Error reading image file:', err);
    return;
  }

  // Convert the image to base64 and log it
  const base64Image = data.toString('base64');
  console.log('data:image/jpeg;base64,' + base64Image);
});
