import os
import tempfile
import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import time
import base64
import json
from io import BytesIO
import streamlit.components.v1 as components

os.environ["STREAMLIT_CACHE_DIR"] = "D:/streamlit_cache"
tempfile.tempdir = "D:/temp"
os.makedirs("D:/temp", exist_ok=True)
os.makedirs("D:/streamlit_cache", exist_ok=True)

# Setting page layout
st.set_page_config(
    page_title="Fishial Recognition",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🐠 Fishial Recognition App")

# Initialize session state for tracking
if 'tracking_paused' not in st.session_state:
    st.session_state.tracking_paused = False
if 'lost_fish' not in st.session_state:
    st.session_state.lost_fish = []
if 'paused_frame' not in st.session_state:
    st.session_state.paused_frame = None
if 'current_frame_idx' not in st.session_state:
    st.session_state.current_frame_idx = 0
if 'trackers' not in st.session_state:
    st.session_state.trackers = []
if 'tracking_started' not in st.session_state:
    st.session_state.tracking_started = False
if 'lost_fish_history' not in st.session_state:
    st.session_state.lost_fish_history = {}
if 'grace_period' not in st.session_state:
    st.session_state.grace_period = 10
if 'video_finished' not in st.session_state:
    st.session_state.video_finished = False
if 'manual_pause' not in st.session_state:
    st.session_state.manual_pause = False
if 'selected_fish_id' not in st.session_state:
    st.session_state.selected_fish_id = None
if 'fabric_updates' not in st.session_state:
    st.session_state.fabric_updates = {}
if 'fabric_mode' not in st.session_state:
    st.session_state.fabric_mode = "add"
if 'fish_rectangles' not in st.session_state:
    st.session_state.fish_rectangles = []
if 'fish_counter' not in st.session_state:
    st.session_state.fish_counter = 1
if 'current_lost_fish_index' not in st.session_state:
    st.session_state.current_lost_fish_index = 0
if 'pause_fish_data' not in st.session_state:
    st.session_state.pause_fish_data = []
if 'pause_canvas_key' not in st.session_state:
    st.session_state.pause_canvas_key = 0
if 'reassign_mode' not in st.session_state:
    st.session_state.reassign_mode = False
if 'fish_to_reassign' not in st.session_state:
    st.session_state.fish_to_reassign = None

# ---------- ENHANCED FABRIC.JS COMPONENT WITH RIGHT-CLICK MENU ----------
def create_enhanced_fabric_js_component(frame_image, fish_data, existing_trackers, mode="add", display_width=800, display_height=600, video_width=1920, video_height=1080, lost_fish_mode=False, current_lost_fish=None):
    """
    Create an enhanced Fabric.js interactive canvas with right-click context menu
    """
    # Convert frame to base64 for background image
    pil_img = Image.fromarray(frame_image)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    background_url = f"data:image/jpeg;base64,{img_str}"
    
    # Prepare fish data for JavaScript
    fish_data_json = json.dumps(fish_data)
    
    # Get existing fish IDs for context menu
    existing_ids = [fish_id for fish_id, _, _ in existing_trackers]
    existing_ids_json = json.dumps(existing_ids)
    
    fabric_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/4.5.0/fabric.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}
            #fabric-canvas {{
                border: 2px solid #0074D9;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                cursor: { "crosshair" if mode == "add" else "default" };
            }}
            .mode-indicator {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 12px;
                z-index: 1000;
            }}
            .lost-fish-indicator {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 12px;
                z-index: 1000;
            }}
            .context-menu {{
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                z-index: 10000;
                min-width: 150px;
                display: none;
            }}
            .context-menu-item {{
                padding: 8px 12px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
                font-family: Arial;
                font-size: 12px;
            }}
            .context-menu-item:hover {{
                background: #f0f0f0;
            }}
            .context-menu-header {{
                background: #0074D9;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px 4px 0 0;
            }}
        </style>
    </head>
    <body>
        <div class="mode-indicator">Mode: {mode.upper()}</div>
        {"<div class='lost-fish-indicator'>Draw box for: " + current_lost_fish + "</div>" if lost_fish_mode and current_lost_fish else ""}
        
        <div id="context-menu" class="context-menu">
            <div class="context-menu-header">Reassign ID</div>
            <div id="context-menu-items"></div>
            <div class="context-menu-item" onclick="assignNewId()">+ New ID</div>
        </div>
        
        <canvas id="fabric-canvas" width="{display_width}" height="{display_height}"></canvas>
        
        <script>
            // Initialize Fabric.js canvas
            const canvas = new fabric.Canvas('fabric-canvas', {{
                selection: true,
                preserveObjectStacking: true
            }});
            
            // Scale factors for coordinate conversion
            const scaleX = {display_width} / {video_width};
            const scaleY = {display_height} / {video_height};
            
            // Load background image
            fabric.Image.fromURL('{background_url}', function(img) {{
                img.set({{
                    scaleX: scaleX,
                    scaleY: scaleY,
                    selectable: false,
                    evented: false
                }});
                canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
            }});
            
            let currentFishData = {fish_data_json};
            let existingFishIds = {existing_ids_json};
            let activeTextbox = null;
            let selectedFish = null;
            let currentMode = "{mode}";
            let isDrawing = false;
            let startX, startY;
            let currentRect = null;
            let lostFishMode = {str(lost_fish_mode).lower()};
            let currentLostFish = "{current_lost_fish if current_lost_fish else ''}";
            let contextMenu = document.getElementById('context-menu');
            let contextMenuItems = document.getElementById('context-menu-items');
            let rightClickedFish = null;
            
            // Function to create a fish object on canvas
            function createFishObject(fish, isTemporary = false, isExisting = false) {{
                // Scale coordinates for display
                const displayX = fish.x * scaleX;
                const displayY = fish.y * scaleY;
                const displayWidth = (fish.width || 60) * scaleX;
                const displayHeight = (fish.height || 40) * scaleY;
                
                // Different colors for existing vs new fish
                const fillColor = isExisting ? 'rgba(0, 255, 0, 0.3)' : (isTemporary ? 'rgba(255, 165, 0, 0.3)' : 'rgba(0, 255, 0, 0.3)');
                const strokeColor = isExisting ? '#00FF00' : (isTemporary ? '#FFA500' : '#00FF00');
                
                // Create fish group with rectangle and label
                const rect = new fabric.Rect({{
                    width: displayWidth,
                    height: displayHeight,
                    fill: fillColor,
                    stroke: strokeColor,
                    strokeWidth: 2,
                    originX: 'center',
                    originY: 'center',
                    rx: 5,
                    ry: 5
                }});
                
                const label = new fabric.Text(fish.id, {{
                    fontSize: 12,
                    fill: 'white',
                    fontFamily: 'Arial',
                    fontWeight: 'bold',
                    originX: 'center',
                    originY: 'center',
                    textBackgroundColor: 'rgba(0,0,0,0.7)',
                    padding: 4,
                    top: displayHeight / 2 + 10
                }});
                
                const group = new fabric.Group([rect, label], {{
                    left: displayX,
                    top: displayY,
                    hasControls: false,
                    hasBorders: false,
                    lockUniScaling: true,
                    fishId: fish.id,
                    originalX: fish.x,
                    originalY: fish.y,
                    originalWidth: fish.width || 60,
                    originalHeight: fish.height || 40,
                    isExisting: isExisting
                }});
                
                canvas.add(group);
                return group;
            }}
            
            // Create existing fish objects (from trackers)
            existingFishIds.forEach(fishId => {{
                // For existing trackers, we don't have exact coordinates during pause
                // We'll create placeholder objects that will be updated with actual data
                const existingFish = {{
                    id: fishId,
                    x: 100, // Placeholder - will be positioned properly in actual implementation
                    y: 100,
                    width: 60,
                    height: 40
                }};
                createFishObject(existingFish, false, true);
            }});
            
            // Create initial fish objects from current fish data
            currentFishData.forEach(fish => {{
                createFishObject(fish);
            }});
            
            // Right-click handler for context menu
            canvas.on('mouse:down', function(options) {{
                if (options.e.button === 2 && options.target && options.target.type === 'group') {{
                    options.e.preventDefault();
                    rightClickedFish = options.target;
                    
                    // Position context menu at mouse coordinates
                    const pointer = canvas.getPointer(options.e);
                    contextMenu.style.left = pointer.x + 'px';
                    contextMenu.style.top = pointer.y + 'px';
                    
                    // Populate context menu with existing IDs
                    contextMenuItems.innerHTML = '';
                    existingFishIds.forEach(fishId => {{
                        if (fishId !== rightClickedFish.fishId) {{
                            const menuItem = document.createElement('div');
                            menuItem.className = 'context-menu-item';
                            menuItem.textContent = fishId;
                            menuItem.onclick = function() {{
                                reassignFishId(fishId);
                            }};
                            contextMenuItems.appendChild(menuItem);
                        }}
                    }});
                    
                    // Show context menu
                    contextMenu.style.display = 'block';
                }} else {{
                    // Hide context menu on other clicks
                    contextMenu.style.display = 'none';
                }}
            }});
            
            // Hide context menu when clicking elsewhere
            canvas.on('mouse:down', function(options) {{
                if (!options.target || options.target.type !== 'group') {{
                    contextMenu.style.display = 'none';
                }}
            }});
            
            // Function to reassign fish ID
            function reassignFishId(newId) {{
                if (rightClickedFish) {{
                    const oldId = rightClickedFish.fishId;
                    
                    // Update the fish ID
                    rightClickedFish.item(1).set('text', newId);
                    rightClickedFish.set('fishId', newId);
                    
                    // Send reassignment to Streamlit
                    const message = {{
                        type: 'fish_reassigned',
                        oldId: oldId,
                        newId: newId,
                        x: Math.round(rightClickedFish.originalX),
                        y: Math.round(rightClickedFish.originalY),
                        width: Math.round(rightClickedFish.originalWidth),
                        height: Math.round(rightClickedFish.originalHeight)
                    }};
                    window.parent.postMessage(message, '*');
                    
                    contextMenu.style.display = 'none';
                }}
            }}
            
            // Function to assign new ID
            function assignNewId() {{
                if (rightClickedFish) {{
                    const newId = prompt('Enter new fish ID:', rightClickedFish.fishId);
                    if (newId && newId !== rightClickedFish.fishId) {{
                        const oldId = rightClickedFish.fishId;
                        
                        // Update the fish ID
                        rightClickedFish.item(1).set('text', newId);
                        rightClickedFish.set('fishId', newId);
                        
                        // Send reassignment to Streamlit
                        const message = {{
                            type: 'fish_reassigned',
                            oldId: oldId,
                            newId: newId,
                            x: Math.round(rightClickedFish.originalX),
                            y: Math.round(rightClickedFish.originalY),
                            width: Math.round(rightClickedFish.originalWidth),
                            height: Math.round(rightClickedFish.originalHeight)
                        }};
                        window.parent.postMessage(message, '*');
                    }}
                    contextMenu.style.display = 'none';
                }}
            }}
            
            // [Rest of the existing Fabric.js code remains the same...]
            // Mouse event handlers for rectangle drawing in add mode
            canvas.on('mouse:down', function(options) {{
                if (currentMode === "add" && !options.target) {{
                    isDrawing = true;
                    const pointer = canvas.getPointer(options.e);
                    startX = pointer.x;
                    startY = pointer.y;
                    
                    currentRect = new fabric.Rect({{
                        left: startX,
                        top: startY,
                        width: 0,
                        height: 0,
                        fill: 'rgba(255, 165, 0, 0.3)',
                        stroke: '#FFA500',
                        strokeWidth: 2,
                        selectable: false,
                        evented: false
                    }});
                    
                    canvas.add(currentRect);
                }} else if (options.target && options.target.type === 'group') {{
                    // Handle fish selection in select/edit mode
                    if (currentMode === "select" || currentMode === "edit") {{
                        // Clear previous selection
                        canvas.getObjects().forEach(obj => {{
                            if (obj.type === 'group') {{
                                obj.set('stroke', null);
                            }}
                        }});
                        
                        // Highlight selected fish
                        options.target.set('stroke', '#FF4136');
                        selectedFish = options.target;
                        
                        // Send selection to Streamlit
                        const message = {{
                            type: 'fish_selected',
                            fishId: options.target.fishId
                        }};
                        window.parent.postMessage(message, '*');
                    }}
                }} else {{
                    // Clicked on background - clear selection
                    if (selectedFish && (currentMode === "select" || currentMode === "edit")) {{
                        selectedFish.set('stroke', null);
                        selectedFish = null;
                    }}
                }}
                canvas.renderAll();
            }});
            
            canvas.on('mouse:move', function(options) {{
                if (!isDrawing) return;
                const pointer = canvas.getPointer(options.e);
                
                currentRect.set({{
                    width: Math.abs(pointer.x - startX),
                    height: Math.abs(pointer.y - startY),
                    left: Math.min(pointer.x, startX),
                    top: Math.min(pointer.y, startY)
                }});
                
                canvas.renderAll();
            }});
            
            canvas.on('mouse:up', function(options) {{
                if (isDrawing && currentMode === "add") {{
                    isDrawing = false;
                    
                    // Only create fish if rectangle is large enough
                    if (currentRect.width > 10 && currentRect.height > 10) {{
                        // Calculate center of rectangle
                        const centerX = currentRect.left + currentRect.width / 2;
                        const centerY = currentRect.top + currentRect.height / 2;
                        
                        // Convert to original video coordinates
                        const newX = centerX / scaleX;
                        const newY = centerY / scaleY;
                        const newWidth = currentRect.width / scaleX;
                        const newHeight = currentRect.height / scaleY;
                        
                        // Determine fish ID based on mode
                        let newFishId;
                        if (lostFishMode && currentLostFish) {{
                            // Automatic pause - use the lost fish ID
                            newFishId = currentLostFish;
                        }} else {{
                            // Manual pause - auto-increment
                            newFishId = 'Fish_' + (canvas.getObjects().filter(obj => obj.type === 'group').length + 1);
                        }}
                        
                        // Create the permanent fish object
                        const newFish = {{
                            id: newFishId,
                            x: Math.round(newX),
                            y: Math.round(newY),
                            width: Math.round(newWidth),
                            height: Math.round(newHeight)
                        }};
                        
                        createFishObject(newFish);
                        
                        // Send new fish to Streamlit
                        const message = {{
                            type: 'fish_added',
                            fishId: newFishId,
                            x: Math.round(newX),
                            y: Math.round(newY),
                            width: Math.round(newWidth),
                            height: Math.round(newHeight)
                        }};
                        window.parent.postMessage(message, '*');
                        
                        // If in lost fish mode, automatically switch to next lost fish or exit add mode
                        if (lostFishMode) {{
                            // In lost fish mode, we only allow one box per lost fish
                            setTimeout(() => {{
                                const message = {{
                                    type: 'next_lost_fish'
                                }};
                                window.parent.postMessage(message, '*');
                            }}, 500);
                        }}
                    }}
                    
                    // Remove the temporary rectangle
                    canvas.remove(currentRect);
                    currentRect = null;
                    canvas.renderAll();
                }}
            }});
            
            // Double-click handler for editing fish IDs (ONLY in manual pause)
            canvas.on('mouse:dblclick', function(options) {{
                if (!lostFishMode && currentMode === "edit" && options.target && options.target.type === 'group') {{
                    const fishGroup = options.target;
                    selectedFish = fishGroup;
                    
                    // Remove any existing textbox
                    if (activeTextbox) {{
                        canvas.remove(activeTextbox);
                    }}
                    
                    // Create editable textbox
                    activeTextbox = new fabric.Textbox(fishGroup.fishId, {{
                        left: fishGroup.left - 50,
                        top: fishGroup.top - 30,
                        width: 100,
                        fontSize: 14,
                        fontFamily: 'Arial',
                        fill: '#0074D9',
                        backgroundColor: 'white',
                        borderColor: '#0074D9',
                        padding: 5,
                        hasControls: false,
                        hasBorders: true
                    }});
                    
                    canvas.add(activeTextbox);
                    activeTextbox.enterEditing();
                    activeTextbox.selectAll();
                    
                    // Handle textbox submission
                    activeTextbox.on('editing:exited', function() {{
                        if (activeTextbox.text && activeTextbox.text !== fishGroup.fishId) {{
                            // Update fish ID
                            const oldId = fishGroup.fishId;
                            const newId = activeTextbox.text;
                            
                            // Update the text in the fish group
                            fishGroup.item(1).set('text', newId);
                            fishGroup.set('fishId', newId);
                            
                            // Send update to Streamlit
                            const message = {{
                                type: 'fish_renamed',
                                oldId: oldId,
                                newId: newId
                            }};
                            window.parent.postMessage(message, '*');
                        }}
                        
                        canvas.remove(activeTextbox);
                        activeTextbox = null;
                        canvas.renderAll();
                    }});
                }}
            }});
            
            // Object movement handler
            canvas.on('object:moving', function(options) {{
                if (options.target && options.target.type === 'group') {{
                    options.target.set('stroke', '#0074D9');
                }}
            }});
            
            // Object moved handler - send final position
            canvas.on('object:modified', function(options) {{
                if (options.target && options.target.type === 'group') {{
                    const fishGroup = options.target;
                    
                    // Convert back to original video coordinates
                    const newX = fishGroup.left / scaleX;
                    const newY = fishGroup.top / scaleY;
                    
                    // Send position update to Streamlit
                    const message = {{
                        type: 'fish_moved',
                        fishId: fishGroup.fishId,
                        newX: Math.round(newX),
                        newY: Math.round(newY)
                    }};
                    window.parent.postMessage(message, '*');
                }}
            }});
            
            canvas.renderAll();
            
            // Prevent default context menu
            canvas.wrapperEl.oncontextmenu = function(e) {{ 
                e.preventDefault();
                return false;
            }};
        </script>
    </body>
    </html>
    '''
    
    return fabric_html

# ---------- ENHANCED MESSAGE HANDLER ----------
def create_enhanced_message_handler():
    """Create a component to handle messages from enhanced Fabric.js"""
    handler_html = '''
    <script>
    // Listen for messages from Fabric.js
    window.addEventListener('message', function(event) {
        // Only process messages from our Fabric.js component
        if (event.data.type && event.data.type.startsWith('fish_')) {
            // Send to Streamlit using the proper method
            const message = {
                'fishial_message': event.data
            };
            window.parent.postMessage(message, '*');
        }
    });
    
    // Hotkey support
    document.addEventListener('keydown', function(event) {
        // A = Apply and Resume
        if (event.key === 'a' || event.key === 'A') {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {
                if (button.innerText.includes('Apply and Resume')) {
                    button.click();
                }
            });
        }
        // R = Resume without applying
        if (event.key === 'r' || event.key === 'R') {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {
                if (button.innerText.includes('Resume Tracking') && !button.innerText.includes('Apply')) {
                    button.click();
                }
            });
        }
        // ESC = Cancel/Close context menus
        if (event.key === 'Escape') {
            // Hide any context menus
            const contextMenus = document.querySelectorAll('.context-menu');
            contextMenus.forEach(menu => {
                menu.style.display = 'none';
            });
        }
        // Enter = Submit form (if in form)
        if (event.key === 'Enter' && event.target.tagName === 'INPUT') {
            setTimeout(() => {
                const buttons = document.querySelectorAll('button');
                buttons.forEach(button => {
                    if (button.innerText.includes('Apply Tracking Boxes') || button.innerText.includes('Apply and Resume')) {
                        button.click();
                    }
                });
            }, 100);
        }
    });
    </script>
    '''
    return handler_html

def scroll_to_top():
    """Scroll the page to top using JavaScript"""
    components.html(
        """
        <script>
        setTimeout(() => {
            const mainContainer = window.parent.document.querySelector('.main');
            if (mainContainer) {
                mainContainer.scrollTo(0, 0);
            }
            const iframeContainer = window.parent.document.querySelector('.stApp');
            if (iframeContainer) {
                iframeContainer.scrollTo(0, 0);
            }
            window.parent.scrollTo(0, 0);
        }, 100);
        </script>
        """,
        height=0
    )

# ---------- WEIGHT SELECTION ----------
st.sidebar.header("Model Settings")

weights_dir = "weights"
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

available_weights = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
if not available_weights:
    st.sidebar.warning("⚠️ No YOLO weight files found in /weights.")
    selected_weight = None
else:
    selected_weight = st.sidebar.selectbox("Select YOLO weights file", available_weights)
    st.sidebar.success(f"Selected: {selected_weight}")

model = None
if selected_weight:
    model_path = os.path.join(weights_dir, selected_weight)
    model = YOLO(model_path)
    st.sidebar.info("✅ Model loaded successfully!")

st.sidebar.subheader("🎥 Select a Video from Repository")

videos_dir = "videos"
if not os.path.exists(videos_dir):
    os.makedirs(videos_dir)

available_videos = [f for f in os.listdir(videos_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

if not available_videos:
    st.sidebar.warning("⚠️ No videos found in the 'videos/' folder. Please add some.")
    st.stop()

selected_video = st.sidebar.selectbox("Choose a video file", available_videos)
video_path = os.path.join(videos_dir, selected_video)

st.success(f"🎞️ Selected video: {selected_video}")

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, first_frame = cap.read()
if not ret:
    st.error("Unable to read video.")
    st.stop()

display_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# ---------- MULTI-FISH SELECTION WITH RECTANGLE DRAWING ----------
if not st.session_state.tracking_paused and not st.session_state.tracking_started:
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    display_width = 800
    display_height = int(height * display_width / width)
    scale_x = width / display_width
    scale_y = height / display_height

    # Instructions for rectangle drawing
    st.info("🔹 **Draw rectangles**: Click and drag to draw rectangles around each fish")
    st.info("🔹 **Auto-naming**: Fish will be automatically named (Fish_1, Fish_2, etc.)")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFA500",
        background_image=frame_pil,
        height=display_height,
        width=display_width,
        drawing_mode="rect",
        update_streamlit=True,
        key="canvas",
    )

    fish_rectangles = []
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        for i, obj in enumerate(canvas_result.json_data["objects"]):
            if obj["type"] == "rect":
                x_canvas, y_canvas = int(obj["left"]), int(obj["top"])
                w_canvas, h_canvas = int(obj["width"]), int(obj["height"])
                
                # Convert to original video coordinates
                x, y = int(x_canvas * scale_x), int(y_canvas * scale_y)
                w, h = int(w_canvas * scale_x), int(h_canvas * scale_y)
                
                # Calculate center point for tracking
                cx, cy = x + w//2, y + h//2
                
                fish_rectangles.append({
                    'id': f"Fish_{i+1}",
                    'x': cx,
                    'y': cy,
                    'width': w,
                    'height': h
                })
        
        st.success(f"Selected {len(fish_rectangles)} fish with rectangle bounding boxes.")

    if fish_rectangles and not st.session_state.tracking_started:
        st.session_state.fish_rectangles = fish_rectangles
        
        if st.button("Start Tracking", type="primary"):
            st.session_state.tracking_started = True
            st.session_state.lost_fish_history = {}
            st.session_state.video_finished = False
            st.rerun()

# ---------- TRACKING EXECUTION WITH MANUAL PAUSE ----------
if st.session_state.tracking_started and not st.session_state.tracking_paused:
    st.subheader("Tracking multiple fish...")
    
    # CONTROL BUTTONS - Always visible during tracking
    control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 1])
    
    with control_col1:
        if st.button("⏸️ Pause Tracking", type="secondary", use_container_width=True):
            st.session_state.manual_pause = True
            st.session_state.tracking_paused = True
            st.session_state.fabric_mode = "add"
            
            # Increment canvas key for fresh canvas
            st.session_state.pause_canvas_key = st.session_state.get('pause_canvas_key', 0) + 1
            
            # Capture current frame for manual editing
            cap_temp = cv2.VideoCapture(video_path)
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
            ret, manual_pause_frame = cap_temp.read()
            cap_temp.release()
            
            if ret:
                st.session_state.paused_frame = manual_pause_frame
                st.session_state.current_frame_idx = st.session_state.frame_idx
                st.session_state.lost_fish = []
                st.rerun()
        
    # Initialize trackers if not already done
    if not st.session_state.trackers:
        trackers = []
        for fish_data in st.session_state.fish_rectangles:
            fish_id = fish_data['id']
            x, y = fish_data['x'], fish_data['y']
            w, h = fish_data['width'], fish_data['height']
            
            tracker = cv2.TrackerCSRT_create()
            
            # Use the rectangle bounding box directly
            bbox = (x - w//2, y - h//2, w, h)
            
            tracker.init(first_frame, bbox)
            trackers.append((fish_id, tracker, bbox))
        
        st.session_state.trackers = trackers
        st.session_state.records = []
        st.session_state.frame_idx = 0

    # Start tracking - use a single continuous loop
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    
    progress_bar = st.progress(0)
    stframe = st.empty()
    status_text = st.empty()
    
    # Track continuously until finished or paused
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.session_state.video_finished = True
            st.session_state.tracking_started = False
            break

        lost_in_current_frame = []
        frame_copy = frame.copy()

        # Update trackers and check for lost fish
        for (fish_id, tracker, bbox) in st.session_state.trackers:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2
                st.session_state.records.append([st.session_state.frame_idx, fish_id, cx, cy, w, h])
                
                # Draw rectangle (original bounding box)
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame_copy, fish_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # If fish was previously lost but now recovered, remove from history
                if fish_id in st.session_state.lost_fish_history:
                    del st.session_state.lost_fish_history[fish_id]
                    
            else:
                lost_in_current_frame.append(fish_id)
                # Update lost duration for this fish
                if fish_id in st.session_state.lost_fish_history:
                    st.session_state.lost_fish_history[fish_id] += 1
                else:
                    st.session_state.lost_fish_history[fish_id] = 1
                
                # Show warning but continue tracking
                lost_duration = st.session_state.lost_fish_history[fish_id] / fps
                cv2.putText(frame_copy, f"{fish_id} lost ({lost_duration:.1f}s)", 
                           (50, 50 + len(lost_in_current_frame)*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # Check if any fish have been lost for longer than grace period
        fish_to_pause_for = []
        for fish_id, lost_frames in st.session_state.lost_fish_history.items():
            lost_seconds = lost_frames / fps
            if lost_seconds >= st.session_state.grace_period:
                fish_to_pause_for.append(fish_id)

        # If any fish exceeded grace period, pause tracking
        if fish_to_pause_for:
            st.session_state.tracking_paused = True
            st.session_state.manual_pause = False
            st.session_state.lost_fish = fish_to_pause_for
            st.session_state.paused_frame = frame_copy
            st.session_state.current_frame_idx = st.session_state.frame_idx
            st.session_state.fabric_mode = "add"
            st.session_state.current_lost_fish_index = 0
            
            # Increment canvas key for fresh canvas
            st.session_state.pause_canvas_key = st.session_state.get('pause_canvas_key', 0) + 1
            
            cap.release()

            scroll_to_top()

            st.rerun()
            break

        # Display current frame with status
        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        stframe.image(rgb, channels="RGB")
        
        # Update status text
        if lost_in_current_frame:
            status_text.warning(f"⚠️ Tracking issues with: {', '.join(lost_in_current_frame)} - Continuing for {st.session_state.grace_period}s grace period")
        else:
            status_text.info("✅ Tracking normally")
        
        # Update progress
        progress_bar.progress(min(st.session_state.frame_idx / frame_count, 1.0))
        
        st.session_state.frame_idx += 1

    cap.release()


# ---------- ENHANCED PAUSE STATE WITH BOUNDING BOXES AND RIGHT-CLICK MENU ----------
# ---------- FABRIC.JS INTERACTIVE PAUSE STATE (WORKING VERSION) ----------
if st.session_state.tracking_paused and st.session_state.paused_frame is not None:
    # Convert paused frame for display
    paused_frame_rgb = cv2.cvtColor(st.session_state.paused_frame, cv2.COLOR_BGR2RGB)
    
    # Create a copy to draw rectangles on existing fish
    paused_frame_with_rectangles = paused_frame_rgb.copy()
    
    # Draw rectangles around existing fish
    for (fish_id, tracker, bbox) in st.session_state.trackers:
        (x, y, w, h) = [int(v) for v in bbox]
        
        # Draw rectangle around existing fish (magenta color)
        cv2.rectangle(paused_frame_with_rectangles, (x, y), (x+w, y+h), (255, 0, 255), 3)
        
        # Add fish ID text
        cv2.putText(paused_frame_with_rectangles, fish_id, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Display status and instructions based on pause type
    if st.session_state.manual_pause:
        st.warning("⏸️ Tracking Manually Paused - Draw NEW rectangles and rename as needed")
    else:
        # Automatic pause - show current lost fish
        current_lost_fish = st.session_state.lost_fish[st.session_state.current_lost_fish_index] if st.session_state.lost_fish else None
        st.error(f"🔴 Tracking Auto-Paused - Draw rectangle for lost fish: **{current_lost_fish}**")
    
    # CREATE SIDEBAR WITH EXISTING FISH IDs
    with st.sidebar:
        st.subheader("📋 Existing Fish IDs")
        st.markdown("Click on a fish ID below to assign it to a new bounding box:")
        
        # Get list of existing fish IDs from trackers
        existing_fish_ids = [fish_id for fish_id, _, _ in st.session_state.trackers]
        
        if existing_fish_ids:
            # Create a container for fish ID buttons
            fish_id_container = st.container()
            
            with fish_id_container:
                # Display each existing fish ID as a clickable button
                for fish_id in existing_fish_ids:
                    if st.button(
                        f"🐟 {fish_id}",
                        key=f"assign_{fish_id}",
                        use_container_width=True,
                        type="secondary" if not (not st.session_state.manual_pause and fish_id == current_lost_fish) else "primary"
                    ):
                        # Store the selected fish ID for reassignment
                        st.session_state.selected_fish_id = fish_id
                        st.info(f"Selected: {fish_id}. Now draw a bounding box to assign this ID.")
        
            # Clear selection button
            if st.session_state.selected_fish_id:
                if st.button("❌ Clear Selection", use_container_width=True):
                    st.session_state.selected_fish_id = None
                    st.rerun()
        else:
            st.info("No existing fish IDs found. All new boxes will create new fish.")
        
        # Show current selection status
        if st.session_state.selected_fish_id:
            st.success(f"**Selected:** {st.session_state.selected_fish_id}")
    
    # CONTROL BUTTONS DURING PAUSE STATE
    st.subheader("Interactive Fish Editor")
    
    # Show legend for visualization
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟣 Magenta rectangles**: Currently tracked fish")
    with col2:
        st.markdown("**🟠 Orange rectangles**: New fish to be added")
    
    st.info("🎨 **Draw rectangles below to add/modify fish tracking**")
    
    # Convert paused frame with rectangles for canvas
    paused_frame_pil = Image.fromarray(paused_frame_with_rectangles)
    
    display_width = 800
    display_height = int(height * display_width / width)
    scale_x = width / display_width
    scale_y = height / display_height
    
    # Use dynamic key to prevent canvas persistence
    canvas_key = f"pause_canvas_{st.session_state.get('pause_canvas_key', 0)}"

    # Use the same canvas component as initial selection
    pause_canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#FFA500",
        background_image=paused_frame_pil,
        height=display_height,
        width=display_width,
        drawing_mode="rect",
        update_streamlit=True,
        key=canvas_key,  # Dynamic key ensures fresh canvas each pause
    )

    # Process rectangles drawn during pause
    if pause_canvas_result.json_data and len(pause_canvas_result.json_data["objects"]) > 0:
        new_rectangles = []
        for i, obj in enumerate(pause_canvas_result.json_data["objects"]):
            if obj["type"] == "rect":
                x_canvas, y_canvas = int(obj["left"]), int(obj["top"])
                w_canvas, h_canvas = int(obj["width"]), int(obj["height"])
                
                # Convert to original video coordinates
                x, y = int(x_canvas * scale_x), int(y_canvas * scale_y)
                w, h = int(w_canvas * scale_x), int(h_canvas * scale_y)
                
                # Calculate center point for tracking
                cx, cy = x + w//2, y + h//2
                
                # Determine fish ID based on selection or context
                if st.session_state.selected_fish_id:
                    # User selected an existing fish ID from sidebar
                    default_id = st.session_state.selected_fish_id
                elif not st.session_state.manual_pause and st.session_state.lost_fish:
                    # Automatic pause for lost fish
                    default_id = st.session_state.lost_fish[st.session_state.current_lost_fish_index] if st.session_state.lost_fish else f"Fish_{len(st.session_state.trackers) + i + 1}"
                else:
                    # Manual pause - create new fish
                    default_id = f"Fish_{len(st.session_state.trackers) + i + 1}"
                
                new_rectangles.append({
                    'id': default_id,
                    'x': cx,
                    'y': cy,
                    'width': w,
                    'height': h,
                    'is_reassignment': st.session_state.selected_fish_id is not None
                })
        
        if new_rectangles:
            st.success(f"Ready to add/update {len(new_rectangles)} fish")
            
            with st.form(key="apply_tracking_boxes_form"):
                temp_fish_data = []
                warning_shown = False
                
                for i, rect_data in enumerate(new_rectangles):
                    # Check if this is reassigning an existing fish
                    existing_ids = [fish_id for fish_id, _, _ in st.session_state.trackers]
                    
                    # Add a warning if user is about to override an existing fish
                    if rect_data['id'] in existing_ids and rect_data['id'] != st.session_state.selected_fish_id:
                        if not warning_shown:
                            st.warning(f"⚠️ Warning: Fish ID '{rect_data['id']}' already exists. This will override the existing tracker.")
                            warning_shown = True
                    
                    # Allow user to confirm or change the fish ID
                    fish_id = st.text_input(
                        f"Fish ID {i+1}",
                        value=rect_data['id'],
                        key=f"fish_id_{i}",
                        placeholder="Enter fish ID"
                    )
                    
                    temp_fish_data.append({
                        'id': fish_id if fish_id else rect_data['id'],
                        'x': rect_data['x'],
                        'y': rect_data['y'], 
                        'width': rect_data['width'],
                        'height': rect_data['height']
                    })
                
                # Show confirmation for reassignment
                if st.session_state.selected_fish_id and new_rectangles:
                    st.warning(f"**Reassignment in progress:** New bounding box will be assigned to **{st.session_state.selected_fish_id}**")
                
                # ENHANCED BUTTONS WITH "APPLY AND RESUME"
                button_col1, button_col2 = st.columns([2, 1])
                with button_col1:
                    apply_resume = st.form_submit_button(
                        "✅ Apply & Resume Tracking", 
                        type="primary",
                        use_container_width=True
                    )
                with button_col2:
                    clear_clicked = st.form_submit_button("🔄 Clear", use_container_width=True)
                
                if apply_resume:
                    # Apply tracking boxes
                    for fish_data in temp_fish_data:
                        fish_id = fish_data['id']
                        x, y = fish_data['x'], fish_data['y']
                        w, h = fish_data['width'], fish_data['height']
                        
                        # Initialize new tracker
                        cap_temp = cv2.VideoCapture(video_path)
                        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
                        ret, recovery_frame = cap_temp.read()
                        cap_temp.release()
                        
                        if ret:
                            new_tracker = cv2.TrackerCSRT_create()
                            bbox = (x - w//2, y - h//2, w, h)
                            
                            # Check if this fish ID already exists
                            existing_index = -1
                            for i, (f_id, tracker, old_bbox) in enumerate(st.session_state.trackers):
                                if f_id == fish_id:
                                    existing_index = i
                                    break
                            
                            if existing_index >= 0:
                                # Overwrite existing tracker
                                st.session_state.trackers[existing_index] = (fish_id, new_tracker, bbox)
                                st.success(f"✅ Updated tracker for: {fish_id}")
                            else:
                                # Add new tracker
                                st.session_state.trackers.append((fish_id, new_tracker, bbox))
                                st.success(f"✅ Added new fish: {fish_id}")
                            
                            new_tracker.init(recovery_frame, bbox)
                    
                    # For automatic pause, move to next lost fish
                    if not st.session_state.manual_pause and st.session_state.lost_fish:
                        if st.session_state.current_lost_fish_index < len(st.session_state.lost_fish) - 1:
                            st.session_state.current_lost_fish_index += 1
                            st.info(f"✅ Box applied. Next lost fish: {st.session_state.lost_fish[st.session_state.current_lost_fish_index]}")
                        else:
                            st.success("✅ All lost fish recovered!")
                    
                    # Clear selection and resume tracking
                    st.session_state.selected_fish_id = None
                    st.session_state.tracking_paused = False
                    st.session_state.manual_pause = False
                    st.session_state.lost_fish = []
                    st.session_state.lost_fish_history = {}
                    st.session_state.paused_frame = None
                    st.session_state.current_lost_fish_index = 0
                    st.session_state.pause_canvas_key = st.session_state.get('pause_canvas_key', 0) + 1
                    scroll_to_top()
                    time.sleep(0.5)
                    
                    st.rerun()
                
                if clear_clicked:
                    st.session_state.selected_fish_id = None
                    st.rerun()

# Manual controls for automatic pause
if st.session_state.tracking_paused and not st.session_state.manual_pause and st.session_state.lost_fish:
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.current_lost_fish_index < len(st.session_state.lost_fish) - 1:
            if st.button("Skip to Next Lost Fish"):
                st.session_state.current_lost_fish_index += 1
                st.rerun()
    
    with col2:
        if st.button("Mark All Lost Fish as Recovered"):
            st.session_state.lost_fish = []
            st.session_state.lost_fish_history = {}
            st.success("✅ All lost fish marked as recovered")

# Resume buttons (ONLY when paused)
if st.session_state.tracking_paused:
   
    if st.button("🔄 Restart Tracking", type="secondary", use_container_width=True):
        # Complete reset
        st.session_state.tracking_started = False
        st.session_state.tracking_paused = False
        st.session_state.manual_pause = False
        st.session_state.trackers = []
        st.session_state.records = []
        st.session_state.lost_fish_history = {}
        st.session_state.video_finished = False
        st.session_state.frame_idx = 0
        st.session_state.paused_frame = None
        st.session_state.pause_canvas_key = 0
        scroll_to_top()
        time.sleep(0.5)
        st.rerun()

# ---------- FINALIZE AND SAVE RESULTS ----------
if st.session_state.video_finished and st.session_state.get('records', []):
    st.subheader("Tracking Complete!")
    
    # Save tracking results
    df = pd.DataFrame(st.session_state.records, columns=["frame", "fish_id", "x", "y", "width", "height"])
    csv_path = "multi_fish_tracking.csv"
    df.to_csv(csv_path, index=False)
    st.success(f"✅ Tracking complete! Data saved to {csv_path}")

    st.dataframe(df.head())
    st.download_button("Download CSV", df.to_csv(index=False), file_name=csv_path, mime="text/csv")
    
    # Reset tracking state
    if st.button("Start New Tracking Session"):
        st.session_state.tracking_started = False
        st.session_state.tracking_paused = False
        st.session_state.manual_pause = False
        st.session_state.trackers = []
        st.session_state.records = []
        st.session_state.lost_fish_history = {}
        st.session_state.video_finished = False
        st.session_state.frame_idx = 0
        st.session_state.selected_fish_id = None
        st.session_state.fabric_updates = {}
        st.session_state.fabric_mode = "select"
        st.session_state.fish_rectangles = []
        st.session_state.fish_counter = 1
        st.session_state.current_lost_fish_index = 0
        st.rerun()