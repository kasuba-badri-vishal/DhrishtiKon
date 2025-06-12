# Grounding Annotation Tool

A web-based tool for annotating images with bounding boxes at block, line, word and point levels for visual question-answering tasks.

## Features

- Load images and their associated question-answer pairs
- Draw bounding boxes and points at four levels:
  - Block level (red boxes)
  - Line level (blue boxes)
  - Word level (green boxes)
  - Point level (orange markers)
- Automatic OCR to generate word-level boxes from line-level annotations
- Modify existing annotations:
  - Select boxes/points by clicking on them
  - Move boxes/points by dragging them
  - Resize boxes using handles
  - Delete boxes/points using the Delete key
- Zoom functionality for precise annotations:
  - Zoom in/out using buttons or keyboard shortcuts
  - Pan around the zoomed image
- Comprehensive keyboard shortcuts for efficient workflow
- Persistent storage of annotations between server restarts
- Clear annotations for specific levels
- Auto-save after any modification

## Installation and Setup

1. Make sure you have Node.js installed on your system
2. Navigate to the code directory
3. Run the server:
   ```
   node server.js
   ```
4. Open your browser and go to http://localhost:3000

## How to Use

1. Select an image from the dropdown menu
2. The image will be displayed along with its question-answer pairs
3. Click on a question-answer pair to select it for annotation
4. Choose the annotation level (block, line, word, or point)
5. Draw annotations on the image:
   - For boxes (block, line, word): click and drag to create a rectangle
   - For points: simply click where you want to place a point
6. To modify an existing annotation:
   - Click on a box or point to select it
   - For boxes: drag the box to move it or use the white resize handles to change dimensions
   - For points: drag to move them
   - Press Delete key to remove the selected annotation
7. To use automatic OCR for word-level boxes:
   - Draw line-level boxes around text lines
   - Click the "Generate Word Boxes from Lines" button or press 'O'
   - The tool will automatically detect words within each line box
   - Results will be displayed as word-level boxes (green)
8. To zoom in/out:
   - Use the zoom buttons in the sidebar
   - Use keyboard shortcuts: '+' to zoom in, '-' to zoom out, '0' to reset
   - Hold Ctrl/Cmd + scroll wheel
9. Use the "Clear Current" button to remove all boxes of the current level for the selected QA pair
10. Click "Save Annotations" to manually save (annotations are also auto-saved after modifications)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| B | Switch to Block level |
| L | Switch to Line level |
| W | Switch to Word level |
| P | Switch to Point level |
| O | Run OCR on line boxes |
| + / = | Zoom in |
| - | Zoom out |
| 0 | Reset zoom to 100% |
| S | Save annotations |
| C | Clear annotations for current level |
| Delete | Delete selected annotation |

## Annotation Structure

The tool saves annotations in the following JSON format:

```json
{
  "image_name.png": [
    {
      "question": "What is shown in the document?",
      "answer": "A certificate",
      "blockBoxes": [
        { "x": 100, "y": 150, "width": 300, "height": 200 }
      ],
      "lineBoxes": [
        { "x": 120, "y": 160, "width": 260, "height": 30 }
      ],
      "wordBoxes": [
        { "x": 120, "y": 160, "width": 80, "height": 30 },
        { "x": 210, "y": 160, "width": 70, "height": 30 }
      ],
      "points": [
        { "x": 150, "y": 200 },
        { "x": 250, "y": 300 }
      ]
    }
  ]
}
```

## Requirements

- Modern web browser with JavaScript enabled
- Node.js for running the server

## OCR Functionality

The OCR functionality uses Tesseract.js to automatically detect words within line-level bounding boxes. To use this feature:

1. First annotate text lines with line-level bounding boxes (blue)
2. Click the "Generate Word Boxes from Lines" button or press 'O'
3. The tool will process each line box, detect words, and create word-level boxes
4. After processing, the tool automatically switches to word-level view to show the results
5. You can still modify the automatically generated word boxes if needed

## Point Annotations

Point annotations are useful for marking specific locations on the document where exact coordinates are needed instead of bounding boxes. To use point annotations:

1. Select the "Point" annotation level
2. Click anywhere on the image to place a point
3. Click on an existing point to select it, then:
   - Drag to move it
   - Press Delete to remove it
4. Points are displayed as orange markers and stored as x,y coordinates in the JSON data

## Troubleshooting

If you're having issues with point annotations:

1. **Points not appearing**: Check that you've selected "Point" as the annotation level before clicking
2. **Points not saving**:
   - Open your browser's developer console (F12 or right-click → Inspect → Console)
   - Look for log messages when clicking to place points and when saving
   - The console will show information about points being created and saved
3. **Visibility**: Point annotations include coordinates labels for easy identification
4. **Data persistence**: Annotations are saved to `grounding_annotations.json` in the code directory (not to `final_annotations.json`)

## Data Persistence

All annotations are automatically saved to the server in `grounding_annotations.json` after any modifications. This file is loaded when the server starts, ensuring that your annotations persist between server restarts.

The server will create a backup of the annotations file each time changes are saved, which can be found at `grounding_annotations.json.backup` in case of data loss. 