
// Paths to the final directory and the final annotations file
IMAGES_PATH = '/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/'
QnA_ANNOTATIONS_PATH = '/data/BADRI/FINAL/THESIS/GRVQA/main/GR-VQA-Grounding/code/annotation/grounding_annotations.json'



const path = require('path');
// Default configuration
const config = {
  PORT: process.env.PORT || 3000,
  ANNOTATIONS_FILE: path.join(__dirname, process.env.ANNOTATIONS_FILE || 'grounding_annotations.json'),
  
  // Base path for the final directory (absolute path)
  FINAL_DIRECTORY: process.env.FINAL_DIRECTORY || IMAGES_PATH,
  
  // Absolute path to the final annotations file
  FINAL_ANNOTATIONS_FILE: process.env.FINAL_ANNOTATIONS_FILE || QnA_ANNOTATIONS_PATH
};

// Ensure paths are absolute
if (!path.isAbsolute(config.FINAL_DIRECTORY)) {
  console.warn('Warning: FINAL_DIRECTORY should be an absolute path');
}

if (!path.isAbsolute(config.FINAL_ANNOTATIONS_FILE)) {
  console.warn('Warning: FINAL_ANNOTATIONS_FILE should be an absolute path');
}

module.exports = config; 