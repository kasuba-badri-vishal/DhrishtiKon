const http = require('http');
const fs = require('fs');
const path = require('path');
const config = require('./config');

const PORT = config.PORT;
const ANNOTATIONS_FILE = config.ANNOTATIONS_FILE;

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif'
};

// Ensure annotations file exists with proper structure
function initializeAnnotationsFile() {
  try {
    if (!fs.existsSync(ANNOTATIONS_FILE)) {
      console.log('Annotations file does not exist. Creating a new one...');
      fs.writeFileSync(ANNOTATIONS_FILE, '{}');
      console.log('Created empty annotations file at:', ANNOTATIONS_FILE);
    } else {
      console.log('Annotations file exists at:', ANNOTATIONS_FILE);
      // Validate JSON structure
      try {
        const data = fs.readFileSync(ANNOTATIONS_FILE, 'utf8');
        JSON.parse(data); // This will throw if invalid JSON
        console.log('Annotations file has valid JSON structure');
      } catch (jsonError) {
        console.error('Annotations file contains invalid JSON. Creating backup and resetting...');
        const backupPath = `${ANNOTATIONS_FILE}.backup-${Date.now()}`;
        fs.copyFileSync(ANNOTATIONS_FILE, backupPath);
        console.log('Created backup at:', backupPath);
        fs.writeFileSync(ANNOTATIONS_FILE, '{}');
        console.log('Reset annotations file with empty JSON object');
      }
    }
  } catch (error) {
    console.error('Error initializing annotations file:', error);
  }
}

// Initialize annotations file at startup
initializeAnnotationsFile();

const server = http.createServer((req, res) => {
  console.log(`Request for ${req.url}`);
  
  // Handle GET request to load annotations
  if (req.method === 'GET' && req.url === '/load-annotations') {
    try {
      if (fs.existsSync(ANNOTATIONS_FILE)) {
        const data = fs.readFileSync(ANNOTATIONS_FILE, 'utf8');
        
        try {
          // Validate JSON before sending
          const parsedData = JSON.parse(data);
          console.log(`Loaded annotations file with ${Object.keys(parsedData).length} images`);
          
          // Look for point annotations specifically
          let pointsFound = false;
          for (const imageName in parsedData) {
            for (const qa of parsedData[imageName]) {
              if (qa.points && qa.points.length > 0) {
                pointsFound = true;
                console.log(`Found ${qa.points.length} points for image ${imageName}`);
                break;
              }
            }
            if (pointsFound) break;
          }
          
          if (!pointsFound) {
            console.log('No point annotations found in the loaded data');
          }
          
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(data);
        } catch (jsonError) {
          console.error('Error parsing annotations JSON:', jsonError);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON in annotations file' }));
        }
      } else {
        console.log('Annotations file not found, sending 404');
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Annotations file not found' }));
      }
    } catch (error) {
      console.error('Error loading annotations:', error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Failed to load annotations' }));
    }
    return;
  }
  
  // Handle POST request to save annotations
  if (req.method === 'POST' && req.url === '/save-annotations') {
    let body = '';
    
    req.on('data', (chunk) => {
      body += chunk.toString();
    });
    
    req.on('end', () => {
      try {
        // Validate JSON
        const parsedData = JSON.parse(body);
        
        // Count total annotations
        let blockCount = 0;
        let lineCount = 0;
        let wordCount = 0;
        let pointCount = 0;
        
        for (const imageName in parsedData) {
          for (const qa of parsedData[imageName]) {
            blockCount += qa.blockBoxes ? qa.blockBoxes.length : 0;
            lineCount += qa.lineBoxes ? qa.lineBoxes.length : 0;
            wordCount += qa.wordBoxes ? qa.wordBoxes.length : 0;
            pointCount += qa.points ? qa.points.length : 0;
          }
        }
        
        console.log(`Saving annotations: ${Object.keys(parsedData).length} images, ${blockCount} blocks, ${lineCount} lines, ${wordCount} words, ${pointCount} points`);
        
        // Create a backup of the existing file before overwriting
        if (fs.existsSync(ANNOTATIONS_FILE)) {
          const backupPath = `${ANNOTATIONS_FILE}.backup`;
          fs.copyFileSync(ANNOTATIONS_FILE, backupPath);
        }
        
        // Write the file
        fs.writeFileSync(ANNOTATIONS_FILE, body);
        console.log('Annotations saved successfully to:', ANNOTATIONS_FILE);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
          success: true,
          stats: {
            images: Object.keys(parsedData).length,
            blocks: blockCount,
            lines: lineCount,
            words: wordCount,
            points: pointCount
          }
        }));
      } catch (error) {
        console.error('Error saving annotations:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: error.message }));
      }
    });
    
    return;
  }
  
  // Handle the root path
  let filePath = req.url === '/' 
    ? path.join(__dirname, 'index.html')
    : path.join(__dirname, req.url);
  
  // If path starts with config.FINAL_DIRECTORY, serve from the configured final directory
  if (req.url.startsWith('/final/')) {
    // Extract the part after /final/
    const relativePath = req.url.substring('/final/'.length);
    
    // Check if FINAL_DIRECTORY is absolute
    if (path.isAbsolute(config.FINAL_DIRECTORY)) {
      filePath = path.join(config.FINAL_DIRECTORY, relativePath);
    } else {
      filePath = path.join(__dirname, '..', config.FINAL_DIRECTORY, relativePath);
    }
  }
  
  // Special case for final_annotations.json
  if (req.url === '/final_annotations.json') {
    // Check if the path is absolute
    if (path.isAbsolute(config.FINAL_ANNOTATIONS_FILE)) {
      filePath = config.FINAL_ANNOTATIONS_FILE;
    } else {
      filePath = path.join(__dirname, '..', config.FINAL_ANNOTATIONS_FILE);
    }
  }

  console.log(`Serving file: ${filePath}`);

  const extname = path.extname(filePath);
  const contentType = MIME_TYPES[extname] || 'text/plain';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // Page not found
        console.error(`File not found: ${filePath}`);
        res.writeHead(404);
        res.end('404 Not Found');
      } else {
        // Server error
        console.error(`Server error: ${err.code}`);
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      }
    } else {
      // Success
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Annotations file path: ${ANNOTATIONS_FILE}`);
  console.log(`Final directory path: ${config.FINAL_DIRECTORY} (${path.isAbsolute(config.FINAL_DIRECTORY) ? 'absolute' : 'relative'})`);
  console.log(`Final annotations file: ${config.FINAL_ANNOTATIONS_FILE} (${path.isAbsolute(config.FINAL_ANNOTATIONS_FILE) ? 'absolute' : 'relative'})`);
  console.log(`URL /final/* will serve from: ${config.FINAL_DIRECTORY}`);
  console.log(`URL /final_annotations.json will serve: ${config.FINAL_ANNOTATIONS_FILE}`);
  console.log(`Press Ctrl+C to stop the server`);
}); 