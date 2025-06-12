document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const imageSelect = document.getElementById('image-select');
    const currentImage = document.getElementById('current-image');
    const annotationCanvas = document.getElementById('annotation-canvas');
    const qaList = document.getElementById('qa-list');
    const saveButton = document.getElementById('save-annotations');
    const clearCurrentButton = document.getElementById('clear-current');
    const zoomInButton = document.getElementById('zoom-in');
    const zoomOutButton = document.getElementById('zoom-out');
    const zoomResetButton = document.getElementById('zoom-reset');
    const zoomLevelDisplay = document.getElementById('zoom-level');
    const imageContainer = document.querySelector('.image-container');
    const canvasContainer = document.querySelector('.canvas-container');
    const ocrButton = document.getElementById('ocr-button');
    const ocrStatus = document.getElementById('ocr-status');
    
    // Canvas context
    const ctx = annotationCanvas.getContext('2d');
    
    // State variables
    let annotations = {};  // Will store all annotations
    let currentImageName = '';
    let currentQA = null;
    let currentAnnotationLevel = 'block';
    let isDrawing = false;
    let startX, startY;
    let currentBoxes = [];
    let allData = {};  // Will store all QA pairs from JSON
    let selectedBox = null; // For box modification
    let isModifying = false;
    let dragStartX, dragStartY;
    let resizeHandle = null;
    const handleSize = 8; // Size of resize handles
    let zoomLevel = 1; // Current zoom level (1 = 100%)
    const zoomStep = 0.1; // How much to zoom in/out per step
    const minZoom = 0.5; // Minimum zoom level
    const maxZoom = 5; // Maximum zoom level
    let isProcessingOCR = false; // Flag to prevent multiple OCR processes
    const pointRadius = 5; // Radius for point markers
    
    // Initialize Tesseract.js worker
    const tesseractWorker = Tesseract.createWorker();
    let workerReady = false;

    // Initialize the OCR worker
    async function initOCRWorker() {
        try {
            await tesseractWorker.load();
            await tesseractWorker.loadLanguage('eng');
            await tesseractWorker.initialize('eng');
            workerReady = true;
            console.log('OCR Worker initialized');
        } catch (error) {
            console.error('Error initializing OCR worker:', error);
        }
    }
    
    // Call worker initialization
    initOCRWorker();
    
    // Set annotation level based on radio buttons
    document.querySelectorAll('input[name="annotation-level"]').forEach(radio => {
        radio.addEventListener('change', () => {
            currentAnnotationLevel = radio.value;
            selectedBox = null;
            redrawAllBoxes();
        });
    });
    
    // OCR processing function
    async function processOCR() {
        if (!workerReady || isProcessingOCR || !currentImageName || currentQA === null) {
            ocrStatus.textContent = 'OCR worker not ready or no image selected';
            return;
        }
        
        const lineBoxes = annotations[currentImageName][currentQA].lineBoxes;
        if (lineBoxes.length === 0) {
            ocrStatus.textContent = 'No line boxes to process';
            return;
        }
        
        isProcessingOCR = true;
        ocrStatus.textContent = 'Processing OCR...';
        ocrStatus.classList.add('active');
        
        try {
            // Create a temporary canvas for image processing
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            for (let i = 0; i < lineBoxes.length; i++) {
                ocrStatus.textContent = `Processing line ${i+1} of ${lineBoxes.length}...`;
                
                const box = lineBoxes[i];
                
                // Set canvas size to match the line box
                tempCanvas.width = box.width;
                tempCanvas.height = box.height;
                
                // Draw the cropped image portion on the temporary canvas
                tempCtx.drawImage(
                    currentImage,
                    box.x, box.y, box.width, box.height,
                    0, 0, box.width, box.height
                );
                
                // Get image data URL
                const imageData = tempCanvas.toDataURL('image/png');
                
                // Process with Tesseract.js
                const result = await tesseractWorker.recognize(imageData);
                
                // Process words from OCR result
                if (result.data.words && result.data.words.length > 0) {
                    for (const word of result.data.words) {
                        // Convert coordinates to image space
                        const wordBox = {
                            x: box.x + word.bbox.x0,
                            y: box.y + word.bbox.y0,
                            width: word.bbox.x1 - word.bbox.x0,
                            height: word.bbox.y1 - word.bbox.y0
                        };
                        
                        // Only add if the box has a reasonable size
                        if (wordBox.width > 3 && wordBox.height > 3) {
                            annotations[currentImageName][currentQA].wordBoxes.push(wordBox);
                        }
                    }
                }
            }
            
            redrawAllBoxes();
            saveAnnotations();
            ocrStatus.textContent = `Added ${annotations[currentImageName][currentQA].wordBoxes.length} word boxes`;
            
            // Auto-switch to word level to see results
            document.getElementById('word-level').checked = true;
            currentAnnotationLevel = 'word';
            redrawAllBoxes();
        } catch (error) {
            console.error('OCR processing error:', error);
            ocrStatus.textContent = 'OCR processing failed';
        } finally {
            isProcessingOCR = false;
            setTimeout(() => {
                ocrStatus.classList.remove('active');
                ocrStatus.textContent = '';
            }, 3000);
        }
    }
    
    // Load initial data
    async function init() {
        try {
            // Load QA pairs from annotations file
            const qaResponse = await fetch('../final_annotations.json');
            allData = await qaResponse.json();
            
            // Try to load existing annotation data first
            try {
                const annotationsResponse = await fetch('/load-annotations');
                if (annotationsResponse.ok) {
                    annotations = await annotationsResponse.json();
                    console.log('Loaded existing annotations');
                } else {
                    // If no existing annotations, create empty structure
                    createEmptyAnnotations();
                }
            } catch (loadError) {
                console.warn('Could not load existing annotations, creating new structure', loadError);
                createEmptyAnnotations();
            }
            
            // Populate image selector
            Object.keys(allData).forEach(imageName => {
                const option = document.createElement('option');
                option.value = imageName;
                option.textContent = imageName;
                imageSelect.appendChild(option);
            });
            
            // Load first image if available
            if (imageSelect.options.length > 0) {
                imageSelect.selectedIndex = 0;
                loadImage(imageSelect.value);
            }
        } catch (error) {
            console.error('Error loading data:', error);
            alert('Failed to load annotation data. Please check the console for details.');
        }
    }
    
    // Create empty annotations structure
    function createEmptyAnnotations() {
        for (const imageName in allData) {
            annotations[imageName] = [];
            allData[imageName].forEach((qa, index) => {
                annotations[imageName][index] = {
                    question: qa.question,
                    answer: qa.answer,
                    blockBoxes: [],
                    lineBoxes: [],
                    wordBoxes: [],
                    points: []
                };
            });
        }
    }
    
    // Load the selected image and its QA pairs
    function loadImage(imageName) {
        currentImageName = imageName;
        currentImage.src = `../final/${imageName}`;
        selectedBox = null;
        
        // When image loads, resize the canvas
        currentImage.onload = () => {
            resetZoom(); // Reset zoom level when loading a new image
            resizeCanvas();
        };
        
        // Load QA pairs for this image
        loadQAPairs(imageName);
    }
    
    // Resize canvas to match the image size
    function resizeCanvas() {
        annotationCanvas.width = currentImage.width * zoomLevel;
        annotationCanvas.height = currentImage.height * zoomLevel;
        
        // Apply zoom transform
        imageContainer.style.transform = `scale(${zoomLevel})`;
        
        redrawAllBoxes();
    }
    
    // Set zoom level
    function setZoom(level) {
        zoomLevel = Math.max(minZoom, Math.min(maxZoom, level));
        zoomLevelDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
        resizeCanvas();
    }
    
    // Zoom in
    function zoomIn() {
        setZoom(zoomLevel + zoomStep);
    }
    
    // Zoom out
    function zoomOut() {
        setZoom(zoomLevel - zoomStep);
    }
    
    // Reset zoom
    function resetZoom() {
        setZoom(1);
    }
    
    // Load question-answer pairs for the current image
    function loadQAPairs(imageName) {
        qaList.innerHTML = '';
        
        if (allData[imageName]) {
            allData[imageName].forEach((qa, index) => {
                const qaItem = document.createElement('div');
                qaItem.className = 'qa-item';
                qaItem.dataset.index = index;
                
                const question = document.createElement('div');
                question.className = 'question';
                question.textContent = qa.question;
                
                const answer = document.createElement('div');
                answer.className = 'answer';
                answer.textContent = qa.answer;
                
                qaItem.appendChild(question);
                qaItem.appendChild(answer);
                qaList.appendChild(qaItem);
                
                // Add click event to select this QA pair
                qaItem.addEventListener('click', () => {
                    document.querySelectorAll('.qa-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    qaItem.classList.add('selected');
                    currentQA = index;
                    selectedBox = null;
                    redrawAllBoxes();
                });
            });
            
            // Select the first QA pair by default
            if (qaList.children.length > 0) {
                qaList.children[0].click();
            }
        }
    }
    
    // Draw boxes on canvas
    function redrawAllBoxes() {
        if (!currentImageName || currentQA === null) return;
        
        // Clear canvas
        ctx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
        
        // Set scale based on zoom level
        ctx.save();
        ctx.scale(zoomLevel, zoomLevel);
        
        const qaAnnotations = annotations[currentImageName][currentQA];
        
        // Draw block boxes
        if (qaAnnotations.blockBoxes) {
            qaAnnotations.blockBoxes.forEach((box, index) => {
                const isSelected = selectedBox && 
                                  currentAnnotationLevel === 'block' && 
                                  selectedBox.index === index;
                drawBox(box, 'rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 0, 0.7)', isSelected);
            });
        }
        
        // Draw line boxes
        if (qaAnnotations.lineBoxes) {
            qaAnnotations.lineBoxes.forEach((box, index) => {
                const isSelected = selectedBox && 
                                  currentAnnotationLevel === 'line' && 
                                  selectedBox.index === index;
                drawBox(box, 'rgba(0, 0, 255, 0.3)', 'rgba(0, 0, 255, 0.7)', isSelected);
            });
        }
        
        // Draw word boxes
        if (qaAnnotations.wordBoxes) {
            qaAnnotations.wordBoxes.forEach((box, index) => {
                const isSelected = selectedBox && 
                                  currentAnnotationLevel === 'word' && 
                                  selectedBox.index === index;
                drawBox(box, 'rgba(0, 128, 0, 0.3)', 'rgba(0, 128, 0, 0.7)', isSelected);
            });
        }
        
        // Draw points
        if (qaAnnotations.points) {
            qaAnnotations.points.forEach((point, index) => {
                const isSelected = selectedBox && 
                                  currentAnnotationLevel === 'point' && 
                                  selectedBox.index === index;
                drawPoint(point, isSelected);
            });
        }
        
        // Restore original scale
        ctx.restore();
    }
    
    function drawBox(box, fillColor, strokeColor, isSelected = false) {
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isSelected ? 3 : 2;
        
        ctx.fillRect(box.x, box.y, box.width, box.height);
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        
        // Draw resize handles if box is selected
        if (isSelected) {
            drawResizeHandles(box);
        }
    }
    
    function drawPoint(point, isSelected = false) {
        const radius = isSelected ? pointRadius + 2 : pointRadius;
        
        // Save current state for drawing
        ctx.save();
        
        // Draw outer circle (border)
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? 'rgba(255, 140, 0, 1)' : 'rgba(255, 165, 0, 0.8)';
        ctx.fill();
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = isSelected ? 'white' : 'rgba(255, 140, 0, 1)';
        ctx.stroke();
        
        if (isSelected) {
            // Add a highlight effect
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius + 3, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 140, 0, 0.5)';
            ctx.stroke();
        }
        
        // Restore context
        ctx.restore();
        
        // Draw coordinate label for debugging
        ctx.save();
        ctx.fillStyle = 'black';
        ctx.font = '10px Arial';
        ctx.fillText(`(${Math.round(point.x)},${Math.round(point.y)})`, point.x + radius + 2, point.y);
        ctx.restore();
    }
    
    // Draw resize handles for selected box
    function drawResizeHandles(box) {
        const handles = [
            { x: box.x, y: box.y }, // top-left
            { x: box.x + box.width / 2, y: box.y }, // top-middle
            { x: box.x + box.width, y: box.y }, // top-right
            { x: box.x, y: box.y + box.height / 2 }, // middle-left
            { x: box.x + box.width, y: box.y + box.height / 2 }, // middle-right
            { x: box.x, y: box.y + box.height }, // bottom-left
            { x: box.x + box.width / 2, y: box.y + box.height }, // bottom-middle
            { x: box.x + box.width, y: box.y + box.height } // bottom-right
        ];
        
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        
        handles.forEach(handle => {
            ctx.beginPath();
            ctx.arc(handle.x, handle.y, handleSize / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        });
    }
    
    // Check if a point is inside a box
    function isPointInBox(x, y, box) {
        return x >= box.x && x <= box.x + box.width &&
               y >= box.y && y <= box.y + box.height;
    }
    
    // Check if a point is near another point (for point selection)
    function isNearPoint(x, y, point) {
        const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
        return distance <= pointRadius * 2; // Give a bit more space for easier selection
    }
    
    // Check if a point is on a resize handle
    function getResizeHandle(x, y, box) {
        const handles = [
            { x: box.x, y: box.y, cursor: 'nwse-resize', position: 'tl' }, // top-left
            { x: box.x + box.width / 2, y: box.y, cursor: 'ns-resize', position: 'tm' }, // top-middle
            { x: box.x + box.width, y: box.y, cursor: 'nesw-resize', position: 'tr' }, // top-right
            { x: box.x, y: box.y + box.height / 2, cursor: 'ew-resize', position: 'ml' }, // middle-left
            { x: box.x + box.width, y: box.y + box.height / 2, cursor: 'ew-resize', position: 'mr' }, // middle-right
            { x: box.x, y: box.y + box.height, cursor: 'nesw-resize', position: 'bl' }, // bottom-left
            { x: box.x + box.width / 2, y: box.y + box.height, cursor: 'ns-resize', position: 'bm' }, // bottom-middle
            { x: box.x + box.width, y: box.y + box.height, cursor: 'nwse-resize', position: 'br' } // bottom-right
        ];
        
        for (const handle of handles) {
            const distance = Math.sqrt(Math.pow(x - handle.x, 2) + Math.pow(y - handle.y, 2));
            if (distance <= handleSize) {
                return handle;
            }
        }
        return null;
    }
    
    // Find which box or point is clicked
    function findSelectedBox(x, y) {
        if (!currentImageName || currentQA === null) return null;
        
        const qaAnnotations = annotations[currentImageName][currentQA];
        
        if (currentAnnotationLevel === 'point') {
            // Check points
            if (qaAnnotations.points) {
                for (let i = qaAnnotations.points.length - 1; i >= 0; i--) {
                    if (isNearPoint(x, y, qaAnnotations.points[i])) {
                        return { index: i, type: 'point' };
                    }
                }
            }
            return null;
        }
        
        // For other annotation types, check boxes
        let boxType = `${currentAnnotationLevel}Boxes`;
        let boxes = qaAnnotations[boxType];
        
        // Check from last (top) to first (bottom) as last drawn is visually on top
        for (let i = boxes.length - 1; i >= 0; i--) {
            if (isPointInBox(x, y, boxes[i])) {
                return { index: i, type: currentAnnotationLevel };
            }
        }
        
        return null;
    }
    
    // Convert screen coordinates to image coordinates (accounting for zoom)
    function screenToImageCoords(screenX, screenY) {
        // Get the actual coordinates on the image considering the zoom level
        return {
            x: screenX / zoomLevel,
            y: screenY / zoomLevel
        };
    }
    
    // Handle drawing on canvas
    annotationCanvas.addEventListener('mousedown', (e) => {
        if (currentQA === null) return;
        
        const rect = annotationCanvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        
        // Convert to image coordinates
        const { x, y } = screenToImageCoords(screenX, screenY);
        
        // Check if we're clicking on an existing box or point
        const clickedElement = findSelectedBox(x, y);
        
        if (clickedElement) {
            selectedBox = clickedElement;
            
            if (clickedElement.type === 'point') {
                // For points, just set up for moving
                console.log(`Selected point at index ${clickedElement.index}`);
                isModifying = true;
                dragStartX = x;
                dragStartY = y;
            } else {
                // For boxes, check if clicking on a resize handle
                const boxType = `${selectedBox.type}Boxes`;
                const box = annotations[currentImageName][currentQA][boxType][selectedBox.index];
                resizeHandle = getResizeHandle(x, y, box);
                
                if (resizeHandle) {
                    // Resize operation
                    isModifying = true;
                    dragStartX = x;
                    dragStartY = y;
                } else if (isPointInBox(x, y, box)) {
                    // Move operation
                    isModifying = true;
                    dragStartX = x;
                    dragStartY = y;
                }
            }
            
            redrawAllBoxes();
        } else {
            // Start drawing a new box or adding a point
            selectedBox = null;
            
            if (currentAnnotationLevel === 'point') {
                // For point level, add a point immediately
                addPoint(x, y);
            } else {
                // For other levels, start drawing a box
                isDrawing = true;
                startX = x;
                startY = y;
            }
        }
    });
    
    annotationCanvas.addEventListener('mousemove', (e) => {
        const rect = annotationCanvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        
        // Convert to image coordinates
        const { x, y } = screenToImageCoords(screenX, screenY);
        
        // Update cursor based on what's under it
        if (!isDrawing && !isModifying) {
            const hoveredElement = findSelectedBox(x, y);
            
            if (hoveredElement) {
                if (hoveredElement.type === 'point') {
                    annotationCanvas.style.cursor = 'move';
                } else {
                    const boxType = `${hoveredElement.type}Boxes`;
                    const box = annotations[currentImageName][currentQA][boxType][hoveredElement.index];
                    const handle = getResizeHandle(x, y, box);
                    
                    if (handle) {
                        annotationCanvas.style.cursor = handle.cursor;
                    } else if (isPointInBox(x, y, box)) {
                        annotationCanvas.style.cursor = 'move';
                    }
                }
            } else {
                annotationCanvas.style.cursor = currentAnnotationLevel === 'point' ? 'pointer' : 'crosshair';
            }
        }
        
        if (isDrawing) {
            // Drawing a new box (not applicable for points)
            redrawAllBoxes();
            
            // Save current transform and set for drawing
            ctx.save();
            ctx.scale(zoomLevel, zoomLevel);
            
            // Draw current box
            ctx.fillStyle = currentAnnotationLevel === 'block' ? 'rgba(255, 0, 0, 0.3)' : 
                           (currentAnnotationLevel === 'line' ? 'rgba(0, 0, 255, 0.3)' : 'rgba(0, 128, 0, 0.3)');
            ctx.strokeStyle = currentAnnotationLevel === 'block' ? 'rgba(255, 0, 0, 0.7)' : 
                             (currentAnnotationLevel === 'line' ? 'rgba(0, 0, 255, 0.7)' : 'rgba(0, 128, 0, 0.7)');
            ctx.lineWidth = 2;
            
            const width = x - startX;
            const height = y - startY;
            
            ctx.fillRect(startX, startY, width, height);
            ctx.strokeRect(startX, startY, width, height);
            
            // Restore the transform
            ctx.restore();
            
        } else if (isModifying && selectedBox) {
            if (selectedBox.type === 'point') {
                // Move point
                const point = annotations[currentImageName][currentQA].points[selectedBox.index];
                point.x = x;
                point.y = y;
                redrawAllBoxes();
            } else {
                // Modify box
                const boxType = `${selectedBox.type}Boxes`;
                const box = annotations[currentImageName][currentQA][boxType][selectedBox.index];
                
                if (resizeHandle) {
                    // Resize the box based on handle position
                    const deltaX = x - dragStartX;
                    const deltaY = y - dragStartY;
                    
                    switch (resizeHandle.position) {
                        case 'tl': // top-left
                            box.x += deltaX;
                            box.y += deltaY;
                            box.width -= deltaX;
                            box.height -= deltaY;
                            break;
                        case 'tm': // top-middle
                            box.y += deltaY;
                            box.height -= deltaY;
                            break;
                        case 'tr': // top-right
                            box.y += deltaY;
                            box.width = x - box.x;
                            box.height -= deltaY;
                            break;
                        case 'ml': // middle-left
                            box.x += deltaX;
                            box.width -= deltaX;
                            break;
                        case 'mr': // middle-right
                            box.width = x - box.x;
                            break;
                        case 'bl': // bottom-left
                            box.x += deltaX;
                            box.width -= deltaX;
                            box.height = y - box.y;
                            break;
                        case 'bm': // bottom-middle
                            box.height = y - box.y;
                            break;
                        case 'br': // bottom-right
                            box.width = x - box.x;
                            box.height = y - box.y;
                            break;
                    }
                    
                    // Ensure width and height are positive
                    if (box.width < 0) {
                        box.x += box.width;
                        box.width = -box.width;
                    }
                    if (box.height < 0) {
                        box.y += box.height;
                        box.height = -box.height;
                    }
                    
                } else {
                    // Move the box
                    box.x += x - dragStartX;
                    box.y += y - dragStartY;
                }
            }
            
            dragStartX = x;
            dragStartY = y;
            redrawAllBoxes();
        }
    });
    
    annotationCanvas.addEventListener('mouseup', (e) => {
        if (isDrawing) {
            const rect = annotationCanvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            
            // Convert to image coordinates
            const { x: endX, y: endY } = screenToImageCoords(screenX, screenY);
            
            // Calculate box dimensions (handle negative values)
            const x = Math.min(startX, endX);
            const y = Math.min(startY, endY);
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            
            // Add box to appropriate array if it has a size
            if (width > 5 && height > 5) {
                const box = { x, y, width, height };
                
                if (currentAnnotationLevel === 'block') {
                    annotations[currentImageName][currentQA].blockBoxes.push(box);
                    selectedBox = { index: annotations[currentImageName][currentQA].blockBoxes.length - 1, type: 'block' };
                } else if (currentAnnotationLevel === 'line') {
                    annotations[currentImageName][currentQA].lineBoxes.push(box);
                    selectedBox = { index: annotations[currentImageName][currentQA].lineBoxes.length - 1, type: 'line' };
                } else if (currentAnnotationLevel === 'word') {
                    annotations[currentImageName][currentQA].wordBoxes.push(box);
                    selectedBox = { index: annotations[currentImageName][currentQA].wordBoxes.length - 1, type: 'word' };
                }
                
                // Save after creating a box
                saveAnnotations();
            }
            
            redrawAllBoxes();
        } else if (isModifying) {
            // Save after modifying a box or point
            saveAnnotations();
        }
        
        isDrawing = false;
        isModifying = false;
        resizeHandle = null;
    });
    
    // Handle wheel event for zooming
    canvasContainer.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            if (e.deltaY < 0) {
                zoomIn();
            } else {
                zoomOut();
            }
        }
    });
    
    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Only if we're not in an input field
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            // Level selection
            if (e.key.toLowerCase() === 'b') {
                document.getElementById('block-level').checked = true;
                currentAnnotationLevel = 'block';
                selectedBox = null;
                redrawAllBoxes();
            } else if (e.key.toLowerCase() === 'l') {
                document.getElementById('line-level').checked = true;
                currentAnnotationLevel = 'line';
                selectedBox = null;
                redrawAllBoxes();
            } else if (e.key.toLowerCase() === 'w') {
                document.getElementById('word-level').checked = true;
                currentAnnotationLevel = 'word';
                selectedBox = null;
                redrawAllBoxes();
            } else if (e.key.toLowerCase() === 'p') {
                document.getElementById('point-level').checked = true;
                currentAnnotationLevel = 'point';
                selectedBox = null;
                redrawAllBoxes();
            }
            
            // OCR processing
            else if (e.key.toLowerCase() === 'o') {
                processOCR();
            }
            
            // Zoom controls
            else if (e.key === '+' || e.key === '=') {
                zoomIn();
            } else if (e.key === '-') {
                zoomOut();
            } else if (e.key === '0') {
                resetZoom();
            }
            
            // Save
            else if (e.key.toLowerCase() === 's') {
                e.preventDefault(); // Prevent browser save dialog
                saveAnnotations();
            }
            
            // Clear
            else if (e.key.toLowerCase() === 'c') {
                if (currentImageName && currentQA !== null) {
                    if (currentAnnotationLevel === 'block') {
                        annotations[currentImageName][currentQA].blockBoxes = [];
                    } else if (currentAnnotationLevel === 'line') {
                        annotations[currentImageName][currentQA].lineBoxes = [];
                    } else if (currentAnnotationLevel === 'word') {
                        annotations[currentImageName][currentQA].wordBoxes = [];
                    } else if (currentAnnotationLevel === 'point') {
                        annotations[currentImageName][currentQA].points = [];
                    }
                    
                    selectedBox = null;
                    redrawAllBoxes();
                    saveAnnotations();
                }
            }
            
            // Delete selected box
            else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBox) {
                e.preventDefault(); // Prevent page navigation on Backspace
                
                if (selectedBox.type === 'point') {
                    annotations[currentImageName][currentQA].points.splice(selectedBox.index, 1);
                } else {
                    const boxType = `${selectedBox.type}Boxes`;
                    annotations[currentImageName][currentQA][boxType].splice(selectedBox.index, 1);
                }
                
                selectedBox = null;
                redrawAllBoxes();
                saveAnnotations();
                console.log('Deleted selected element');
            }
        }
    });
    
    // Clear current annotations for current QA and level
    clearCurrentButton.addEventListener('click', () => {
        if (currentImageName && currentQA !== null) {
            if (currentAnnotationLevel === 'block') {
                annotations[currentImageName][currentQA].blockBoxes = [];
            } else if (currentAnnotationLevel === 'line') {
                annotations[currentImageName][currentQA].lineBoxes = [];
            } else if (currentAnnotationLevel === 'word') {
                annotations[currentImageName][currentQA].wordBoxes = [];
            } else if (currentAnnotationLevel === 'point') {
                annotations[currentImageName][currentQA].points = [];
            }
            
            selectedBox = null;
            redrawAllBoxes();
            saveAnnotations();
        }
    });
    
    // Function to save annotations
    async function saveAnnotations() {
        try {
            // Ensure we have empty arrays for all types of annotations if they don't exist
            for (const imageName in annotations) {
                for (let i = 0; i < annotations[imageName].length; i++) {
                    const qa = annotations[imageName][i];
                    if (!qa.blockBoxes) qa.blockBoxes = [];
                    if (!qa.lineBoxes) qa.lineBoxes = [];
                    if (!qa.wordBoxes) qa.wordBoxes = [];
                    if (!qa.points) qa.points = [];
                }
            }
            
            // Convert annotations to JSON
            const json = JSON.stringify(annotations, null, 2);
            
            // Log a sample of annotations for debugging
            if (currentImageName && currentQA !== null) {
                console.log(`Saving annotations for ${currentImageName}, QA #${currentQA}:`);
                console.log(`- Block boxes: ${annotations[currentImageName][currentQA].blockBoxes.length}`);
                console.log(`- Line boxes: ${annotations[currentImageName][currentQA].lineBoxes.length}`);
                console.log(`- Word boxes: ${annotations[currentImageName][currentQA].wordBoxes.length}`);
                console.log(`- Points: ${annotations[currentImageName][currentQA].points.length}`);
                if (annotations[currentImageName][currentQA].points.length > 0) {
                    console.log(`- First point: (${annotations[currentImageName][currentQA].points[0].x}, ${annotations[currentImageName][currentQA].points[0].y})`);
                }
            }
            
            // Send data to the server endpoint
            const response = await fetch('/save-annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: json
            });
            
            if (!response.ok) {
                throw new Error('Failed to save annotations');
            }
            
            console.log('Annotations saved successfully');
        } catch (error) {
            console.error('Error saving annotations:', error);
        }
    }
    
    // Save button click handler
    saveButton.addEventListener('click', saveAnnotations);
    
    // Handle image selection change
    imageSelect.addEventListener('change', () => {
        loadImage(imageSelect.value);
    });
    
    // Zoom button event listeners
    zoomInButton.addEventListener('click', zoomIn);
    zoomOutButton.addEventListener('click', zoomOut);
    zoomResetButton.addEventListener('click', resetZoom);
    
    // OCR button event listener
    ocrButton.addEventListener('click', processOCR);
    
    // Add a point to the current QA pair
    function addPoint(x, y) {
        const point = { x, y };
        
        // Ensure the points array exists
        if (!annotations[currentImageName][currentQA].points) {
            annotations[currentImageName][currentQA].points = [];
        }
        
        // Add the point
        annotations[currentImageName][currentQA].points.push(point);
        
        // Set as selected
        selectedBox = { 
            index: annotations[currentImageName][currentQA].points.length - 1, 
            type: 'point' 
        };
        
        console.log(`Point added at (${x}, ${y}). Total points: ${annotations[currentImageName][currentQA].points.length}`);
        
        // Save and redraw
        saveAnnotations();
        redrawAllBoxes();
    }
    
    // Initialize the application
    init();
}); 