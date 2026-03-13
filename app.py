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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qualitative

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
# NEW: store tracking records
if 'records' not in st.session_state:
    st.session_state.records = []

# NEW: store per-fish trajectory buffers (manual mode)
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = {}
if 'last_traj_plot_frame' not in st.session_state:
    st.session_state.last_traj_plot_frame = -1
if 'tracking_source' not in st.session_state:
    st.session_state.tracking_source = None  # 'manual' or 'yolo'
if 'reference_trajectories' not in st.session_state:
    st.session_state.reference_trajectories = {}
if 'reference_file_names' not in st.session_state:
    st.session_state.reference_file_names = []
if 'reference_metadata' not in st.session_state:
    st.session_state.reference_metadata = {}
if 'reference_file_name' not in st.session_state:
    st.session_state.reference_file_name = None

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

def _coerce_reference_metadata_value(series, key):
    if key in series and len(series[key].dropna()) > 0:
        try:
            return int(float(series[key].dropna().iloc[0]))
        except Exception:
            return None
    return None

def parse_reference_trajectory_file(uploaded_file):
    """Parse uploaded trajectory CSV/JSON into {fish_id: {'x':[], 'y':[], 'frame':[]}}."""
    if uploaded_file is None:
        return {}, {}

    filename = uploaded_file.name
    suffix = os.path.splitext(filename)[1].lower()

    def finalize_df(df, default_fish_id=None):
        if df is None or df.empty:
            return {}, {}
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Normalize common aliases
        rename_map = {}
        for col in df.columns:
            if col in ('time', 'timestamp', 't'):
                rename_map[col] = 'frame'
            elif col in ('cx', 'center_x'):
                rename_map[col] = 'x'
            elif col in ('cy', 'center_y'):
                rename_map[col] = 'y'
            elif col in ('id', 'track_id'):
                rename_map[col] = 'fish_id'
        if rename_map:
            df = df.rename(columns=rename_map)

        if 'fish_id' not in df.columns:
            df['fish_id'] = default_fish_id or os.path.splitext(filename)[0]
        if 'frame' not in df.columns:
            df['frame'] = np.arange(len(df))
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("Reference file must contain x and y columns (or cx/cy aliases).")

        for col in ['x', 'y', 'frame']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['x', 'y']).copy()
        df['frame'] = df['frame'].fillna(method='ffill').fillna(method='bfill').fillna(0)

        metadata = {
            'video_width': _coerce_reference_metadata_value(df, 'video_width'),
            'video_height': _coerce_reference_metadata_value(df, 'video_height'),
            'source_file': filename,
        }

        trajectories = {}
        for fish_id, fish_df in df.groupby('fish_id', dropna=False):
            fish_label = str(fish_id) if pd.notna(fish_id) and str(fish_id).strip() else (default_fish_id or 'Reference_Fish')
            fish_df = fish_df.sort_values('frame')
            trajectories[fish_label] = {
                'x': [int(v) for v in fish_df['x'].tolist()],
                'y': [int(v) for v in fish_df['y'].tolist()],
                'frame': [int(v) for v in fish_df['frame'].tolist()]
            }
        return trajectories, metadata

    if suffix == '.csv':
        df = pd.read_csv(uploaded_file)
        return finalize_df(df)

    if suffix == '.json':
        raw = json.load(uploaded_file)
        metadata = {'source_file': filename}

        if isinstance(raw, dict) and 'trajectories' in raw:
            base = raw.get('trajectories', {})
            metadata.update({k: raw.get(k) for k in ('video_width', 'video_height') if k in raw})
        else:
            base = raw

        if isinstance(base, dict):
            trajectories = {}
            for fish_id, series in base.items():
                if isinstance(series, dict):
                    xs = series.get('x', [])
                    ys = series.get('y', [])
                    frames = series.get('frame', list(range(len(xs))))
                elif isinstance(series, list):
                    xs = [pt.get('x') for pt in series if isinstance(pt, dict)]
                    ys = [pt.get('y') for pt in series if isinstance(pt, dict)]
                    frames = [pt.get('frame', i) for i, pt in enumerate(series) if isinstance(pt, dict)]
                else:
                    continue
                if len(xs) == 0 or len(ys) == 0:
                    continue
                trajectories[str(fish_id)] = {
                    'x': [int(float(v)) for v in xs if v is not None],
                    'y': [int(float(v)) for v in ys if v is not None],
                    'frame': [int(float(v)) for v in frames[:len(xs)] if v is not None]
                }
            return trajectories, metadata

        if isinstance(base, list):
            df = pd.DataFrame(base)
            trajectories, df_metadata = finalize_df(df, default_fish_id=os.path.splitext(filename)[0])
            metadata.update(df_metadata)
            return trajectories, metadata

        raise ValueError('Unsupported JSON trajectory format.')

    raise ValueError('Unsupported file type. Upload a CSV or JSON trajectory file.')


def merge_reference_trajectory_files(uploaded_files):
    """Merge one or more uploaded trajectory files into a single reference overlay dict."""
    merged_trajectories = {}
    merged_metadata = {'source_files': []}

    if not uploaded_files:
        return merged_trajectories, merged_metadata

    for uploaded_file in uploaded_files:
        file_trajectories, file_metadata = parse_reference_trajectory_file(uploaded_file)
        source_name = file_metadata.get('source_file', getattr(uploaded_file, 'name', 'reference'))
        merged_metadata['source_files'].append(source_name)

        for fish_id, series in file_trajectories.items():
            base_id = str(fish_id) if str(fish_id).strip() else 'Reference_Fish'
            candidate_id = base_id
            suffix = 2
            while candidate_id in merged_trajectories:
                candidate_id = f"{base_id}_{suffix}"
                suffix += 1
            merged_trajectories[candidate_id] = series

        for dim_key in ('video_width', 'video_height'):
            value = file_metadata.get(dim_key)
            if value is None:
                continue
            existing = merged_metadata.get(dim_key)
            if existing is None:
                merged_metadata[dim_key] = value
            elif existing != value:
                merged_metadata[dim_key] = 'mixed'

    return merged_trajectories, merged_metadata

# ---------- WEIGHT SELECTION ----------
st.sidebar.header("Model Settings")

st.sidebar.subheader("⚙️ Performance Settings")
tracker_choice = st.sidebar.selectbox(
    "Tracker type",
    options=["CSRT (accurate, slow)", "KCF (faster)", "MOSSE (fastest)"],
    index=0
)
processing_scale = st.sidebar.slider(
    "Processing scale (lower = faster)",
    min_value=0.25, max_value=1.0, value=1.0, step=0.05
)
frame_skip = st.sidebar.number_input(
    "Process every Nth frame", min_value=1, max_value=10, value=1
)


# --- NEW: Tracking mode selection (manual vs YOLO-all) ---
st.sidebar.subheader("🎯 Tracking Mode")
tracking_mode = st.sidebar.radio(
    "How do you want to start tracking?",
    options=[
        "Manual: draw boxes and track selected fish",
        "YOLO: detect and track ALL fish automatically"
    ],
    index=0
)

# YOLO settings only relevant for YOLO-all mode
if tracking_mode.startswith("YOLO"):
    yolo_conf = st.sidebar.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
    yolo_iou = st.sidebar.slider("YOLO IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
    yolo_max_det = st.sidebar.number_input("Max detections", min_value=1, max_value=200, value=30, step=1)
else:
    yolo_conf, yolo_iou, yolo_max_det = 0.25, 0.45, 30

# Trajectory (live XY) settings - only for MANUAL mode
if tracking_mode.startswith("Manual"):
    st.sidebar.subheader("📈 Live Trajectory (Manual mode)")
    show_live_trajectory = st.sidebar.checkbox("Show live XY trajectory", value=True)
    traj_refresh_every = st.sidebar.slider("Trajectory refresh (every N frames)", 1, 60, 10, 1)
else:
    show_live_trajectory = False
    traj_refresh_every = 10

if tracking_mode.startswith("Manual"):
    st.sidebar.subheader("🧭 Reference Trajectory Overlay")
    enable_reference_overlay = st.sidebar.checkbox("Overlay previous trajectory file", value=False)
    reference_upload = st.sidebar.file_uploader(
        "Upload trajectory CSV or JSON",
        type=["csv", "json"],
        accept_multiple_files=False,
        help="Upload a previously saved trajectory file to overlay as a dashed reference path."
    ) if enable_reference_overlay else None

    if enable_reference_overlay and reference_upload is not None:
        try:
            ref_trajectories, ref_metadata = parse_reference_trajectory_file(reference_upload)
            st.session_state.reference_trajectories = ref_trajectories
            st.session_state.reference_metadata = ref_metadata
            st.session_state.reference_file_name = reference_upload.name
            st.sidebar.success(f"Loaded reference file: {reference_upload.name}")
            st.sidebar.caption(f"Reference fish loaded: {len(ref_trajectories)}")
            ref_w = ref_metadata.get('video_width')
            ref_h = ref_metadata.get('video_height')
            if ref_w is not None and ref_h is not None and (ref_w != width or ref_h != height):
                st.sidebar.warning(
                    f"Reference file uses {ref_w}×{ref_h}, while current video is {width}×{height}. "
                    "Overlay will still render, but direct comparison may be misleading if the camera view/resolution differs."
                )
        except Exception as e:
            st.session_state.reference_trajectories = {}
            st.session_state.reference_metadata = {}
            st.session_state.reference_file_name = None
            st.sidebar.error(f"Could not read reference trajectory file: {e}")
    elif not enable_reference_overlay:
        st.session_state.reference_trajectories = {}
        st.session_state.reference_metadata = {}
        st.session_state.reference_file_name = None
else:
    enable_reference_overlay = False
    st.session_state.reference_trajectories = {}
    st.session_state.reference_metadata = {}
    st.session_state.reference_file_name = None

st.session_state.show_live_trajectory = bool(show_live_trajectory)
st.session_state.traj_refresh_every = int(traj_refresh_every)
st.session_state.enable_reference_overlay = bool(enable_reference_overlay)

# Persist in session_state so the tracking loop can read it
st.session_state.tracker_choice = tracker_choice
st.session_state.processing_scale = processing_scale
st.session_state.frame_skip = frame_skip
st.session_state.tracking_mode = tracking_mode
st.session_state.yolo_conf = float(yolo_conf)
st.session_state.yolo_iou = float(yolo_iou)
st.session_state.yolo_max_det = int(yolo_max_det)

def create_tracker(choice):
    if "KCF" in choice:
        return cv2.TrackerKCF_create()
    elif "MOSSE" in choice:
        return cv2.legacy.TrackerMOSSE_create()   # Note: legacy in newer OpenCV
    else:
        return cv2.TrackerCSRT_create()
    
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


# ---------- MULTI-FISH SELECTION / YOLO-ALL START MODE ----------
def yolo_detect_fish_boxes(frame_bgr, model, conf=0.25, iou=0.45, max_det=30):
    """Run YOLO on a BGR frame and return list of bboxes as (x, y, w, h) in pixel ints."""
    if model is None:
        return []
    # Ultralytics accepts numpy arrays directly
    results = model.predict(frame_bgr, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    # xyxy in pixels
    xyxy = r0.boxes.xyxy.cpu().numpy()
    bboxes = []
    for (x1, y1, x2, y2) in xyxy:
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(width - 1, x2)), int(min(height - 1, y2))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        bboxes.append((x1, y1, w, h))
    return bboxes

# Only show selection UI when not started and not paused
if not st.session_state.tracking_paused and not st.session_state.tracking_started:
    tracking_mode = st.session_state.get("tracking_mode", "Manual: draw boxes and track selected fish")

    # --- MODE A: Manual selection (current behavior) ---
    if tracking_mode.startswith("Manual"):
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
        if canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) > 0:
            for i, obj in enumerate(canvas_result.json_data["objects"]):
                if obj.get("type") == "rect":
                    x_canvas, y_canvas = int(obj["left"]), int(obj["top"])
                    w_canvas, h_canvas = int(obj["width"]), int(obj["height"])

                    # Convert to original video coordinates
                    x, y = int(x_canvas * scale_x), int(y_canvas * scale_y)
                    w, h = int(w_canvas * scale_x), int(h_canvas * scale_y)

                    # Calculate center point for tracking
                    cx, cy = x + w // 2, y + h // 2

                    fish_rectangles.append({
                        "id": f"Fish_{i+1}",
                        "x": cx,
                        "y": cy,
                        "width": w,
                        "height": h
                    })

            st.success(f"Selected {len(fish_rectangles)} fish with rectangle bounding boxes.")

        if fish_rectangles and not st.session_state.tracking_started:
            st.session_state.fish_rectangles = fish_rectangles

            if st.button("Start Tracking", type="primary"):
                st.session_state.tracking_source = "manual"
                st.session_state.trajectories = {fd['id']: {'x': [], 'y': [], 'frame': []} for fd in st.session_state.fish_rectangles}
                st.session_state.last_traj_plot_frame = -1
                st.session_state.tracking_started = True
                st.session_state.lost_fish_history = {}
                st.session_state.video_finished = False
                st.session_state.records = []  # clear old records
                st.rerun()

    # --- MODE B: YOLO detects ALL fish on first frame ---
    else:
        st.info("🤖 YOLO mode: the app will detect all fish on the FIRST frame and start tracking them.")
        if model is None:
            st.error("YOLO weights are not loaded. Please select a .pt file in the sidebar to use YOLO mode.")
            st.stop()

        conf = st.session_state.get("yolo_conf", 0.25)
        iou = st.session_state.get("yolo_iou", 0.45)
        max_det = st.session_state.get("yolo_max_det", 30)

        yolo_bboxes = yolo_detect_fish_boxes(first_frame, model, conf=conf, iou=iou, max_det=max_det)

        if not yolo_bboxes:
            st.warning("No fish detections on the first frame with the current YOLO settings. Try lowering confidence.")
        else:
            # Convert YOLO bboxes to your fish_rectangles schema
            fish_rectangles = []
            preview = first_frame.copy()
            for i, (x, y, w, h) in enumerate(yolo_bboxes, start=1):
                cx, cy = x + w // 2, y + h // 2
                fish_id = f"Fish_{i}"
                fish_rectangles.append({"id": fish_id, "x": cx, "y": cy, "width": w, "height": h})
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview, fish_id, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            st.success(f"YOLO detected {len(fish_rectangles)} fish on the first frame.")
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="YOLO detections on first frame", use_column_width=True)

            if st.button("Start Tracking (YOLO)", type="primary"):
                st.session_state.fish_rectangles = fish_rectangles
                st.session_state.tracking_source = "yolo"
                st.session_state.trajectories = {}  # live trajectories are only shown for manual mode
                st.session_state.last_traj_plot_frame = -1
                st.session_state.tracking_started = True
                st.session_state.lost_fish_history = {}
                st.session_state.video_finished = False
                st.session_state.records = []  # clear old records
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
            
            tracker = create_tracker(st.session_state.get('tracker_choice', tracker_choice))
            
            # Use the rectangle bounding box directly
            bbox = (x - w//2, y - h//2, w, h)
            
            tracker.init(first_frame, bbox)
            trackers.append((fish_id, tracker, bbox))
        
        st.session_state.trackers = trackers
        st.session_state.frame_idx = 0

    # In the tracking section, after reading a frame
    # Retrieve performance settings from session state (set via sidebar)
    processing_scale = st.session_state.get('processing_scale', 1.0)
    frame_skip = st.session_state.get('frame_skip', 1)
    tracker_choice = st.session_state.get('tracker_choice', 'CSRT (accurate, slow)')


    def _make_traj_fig(trajectories_dict, reference_dict=None):
        fig = go.Figure()

        reference_dict = reference_dict or {}
        live_palette = (plotly_qualitative.Plotly + plotly_qualitative.Safe + plotly_qualitative.Vivid)
        ref_palette = (plotly_qualitative.Dark24 + plotly_qualitative.Bold + plotly_qualitative.Set1)

        # Static reference trajectories first (solid lines, contrasting colors)
        for idx, (fish_id, series) in enumerate(reference_dict.items()):
            xs = series.get('x', [])
            ys = series.get('y', [])
            if len(xs) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=f"Ref_{fish_id}",
                line=dict(color=ref_palette[idx % len(ref_palette)], width=2), opacity=0.72
            ))

        # Live trajectories on top
        for idx, (fish_id, series) in enumerate(trajectories_dict.items()):
            xs = series.get('x', [])
            ys = series.get('y', [])
            if len(xs) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=fish_id,
                line=dict(color=live_palette[idx % len(live_palette)], width=3), opacity=1.0
            ))
        fig.update_layout(
            title='Trajectory Comparison (XY)',
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        # Lock axes to full video frame (match Matplotlib-style full-frame view)
        fig.update_xaxes(title_text='X (pixels)', range=[0, width], autorange=False)
        # Invert Y to match image coordinate system (origin at top-left)
        fig.update_yaxes(title_text='Y (pixels)', range=[height, 0], autorange=False)
        return fig

    # Create placeholders for display (video + live trajectory)
    progress_bar = st.progress(0)
    video_col, traj_col = st.columns([1.35, 1])
    with video_col:
        frame_placeholder = st.empty()
        status_text = st.empty()
    with traj_col:
        traj_placeholder = st.empty()
        if st.session_state.get('tracking_source') == 'manual' and st.session_state.get('show_live_trajectory', True):
            traj_placeholder.plotly_chart(_make_traj_fig({}, st.session_state.get('reference_trajectories', {})), use_container_width=True)

    # We'll keep last known positions for skipped frames
    last_positions = {fish_id: bbox for fish_id, _, bbox in st.session_state.trackers}


    def _make_traj_fig(trajectories_dict, reference_dict=None):
        fig = go.Figure()

        reference_dict = reference_dict or {}
        live_palette = (plotly_qualitative.Plotly + plotly_qualitative.Safe + plotly_qualitative.Vivid)
        ref_palette = (plotly_qualitative.Dark24 + plotly_qualitative.Bold + plotly_qualitative.Set1)

        # Static reference trajectories first (solid lines, contrasting colors)
        for idx, (fish_id, series) in enumerate(reference_dict.items()):
            xs = series.get('x', [])
            ys = series.get('y', [])
            if len(xs) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=f"Ref_{fish_id}",
                line=dict(color=ref_palette[idx % len(ref_palette)], width=2), opacity=0.72
            ))

        # Live trajectories on top
        for idx, (fish_id, series) in enumerate(trajectories_dict.items()):
            xs = series.get('x', [])
            ys = series.get('y', [])
            if len(xs) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', name=fish_id,
                line=dict(color=live_palette[idx % len(live_palette)], width=3), opacity=1.0
            ))
        fig.update_layout(
            title='Trajectory Comparison (XY)',
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        # Lock axes to full video frame (match Matplotlib-style full-frame view)
        fig.update_xaxes(title_text='X (pixels)', range=[0, width], autorange=False)
        # Invert Y to match image coordinate system (origin at top-left)
        fig.update_yaxes(title_text='Y (pixels)', range=[height, 0], autorange=False)
        return fig

    # Start tracking - use a single continuous loop
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.session_state.video_finished = True
            st.session_state.tracking_started = False
            break

        # --- FRAME SKIPPING ---
        # If this frame is not to be processed, just display last known positions
        if st.session_state.frame_idx % frame_skip != 0:
            # Draw last known bounding boxes on the frame
            frame_copy = frame.copy()
            for fish_id, bbox in last_positions.items():
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame_copy, fish_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 2)

            # Throttle UI updates (update every 5 frames)
            if st.session_state.frame_idx % 5 == 0:
                frame_placeholder.image(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
                # Live trajectory update (manual mode only)
                if (st.session_state.get('tracking_source') == 'manual' and st.session_state.get('show_live_trajectory', True)
                    and st.session_state.frame_idx - st.session_state.get('last_traj_plot_frame', -1) >= st.session_state.get('traj_refresh_every', 10)):
                    traj_placeholder.plotly_chart(_make_traj_fig(st.session_state.trajectories, st.session_state.get('reference_trajectories', {})), use_container_width=True)
                    st.session_state.last_traj_plot_frame = st.session_state.frame_idx

            st.session_state.frame_idx += 1
            progress_bar.progress(min(st.session_state.frame_idx / frame_count, 1.0))
            continue

        # --- PROCESS THIS FRAME ---
        # Scale frame if needed
        if processing_scale != 1.0:
            small_frame = cv2.resize(frame, None, fx=processing_scale, fy=processing_scale)
        else:
            small_frame = frame

        lost_in_current_frame = []
        new_trackers = []  # we'll rebuild tracker list with updated bboxes
        updated_positions = {}  # store new positions for records

        for fish_id, tracker, bbox in st.session_state.trackers:
            # Scale bbox if we scaled the frame
            if processing_scale != 1.0:
                small_bbox = tuple(int(v * processing_scale) for v in bbox)
                success, new_small_bbox = tracker.update(small_frame)
                if success:
                    # Convert back to original coordinates
                    new_bbox = tuple(int(v / processing_scale) for v in new_small_bbox)
                    last_positions[fish_id] = new_bbox  # update last known
                    updated_positions[fish_id] = new_bbox
                    new_trackers.append((fish_id, tracker, new_bbox))
                else:
                    lost_in_current_frame.append(fish_id)
                    new_trackers.append((fish_id, tracker, bbox))  # keep old bbox
            else:
                success, new_bbox = tracker.update(frame)
                if success:
                    last_positions[fish_id] = new_bbox
                    updated_positions[fish_id] = new_bbox
                    new_trackers.append((fish_id, tracker, new_bbox))
                else:
                    lost_in_current_frame.append(fish_id)
                    new_trackers.append((fish_id, tracker, bbox))

        st.session_state.trackers = new_trackers

        # Update lost fish history and check grace period
        for fish_id in lost_in_current_frame:
            st.session_state.lost_fish_history[fish_id] = st.session_state.lost_fish_history.get(fish_id, 0) + 1
        for fish_id in last_positions.keys():
            if fish_id not in lost_in_current_frame:
                st.session_state.lost_fish_history.pop(fish_id, None)  # recovered

        # Record positions for this frame (for all fish, even if lost we keep old bbox)
        for fish_id, bbox in last_positions.items():
            x, y, w, h = bbox
            # Store center coordinates (x + w/2, y + h/2) as x,y
            cx = x + w//2
            cy = y + h//2
            st.session_state.records.append([st.session_state.frame_idx, fish_id, cx, cy, w, h])
            # Update live trajectories (manual mode only)
            if st.session_state.get('tracking_source') == 'manual' and st.session_state.get('show_live_trajectory', True):
                if fish_id not in st.session_state.trajectories:
                    st.session_state.trajectories[fish_id] = {'x': [], 'y': [], 'frame': []}
                st.session_state.trajectories[fish_id]['x'].append(int(cx))
                st.session_state.trajectories[fish_id]['y'].append(int(cy))
                st.session_state.trajectories[fish_id]['frame'].append(int(st.session_state.frame_idx))

        # Check if any fish lost longer than grace period
        fish_to_pause_for = []
        for fish_id, lost_frames in st.session_state.lost_fish_history.items():
            if lost_frames / fps >= st.session_state.grace_period:
                fish_to_pause_for.append(fish_id)

        if fish_to_pause_for:
            # Pause and show lost fish
            st.session_state.tracking_paused = True
            st.session_state.manual_pause = False
            st.session_state.lost_fish = fish_to_pause_for
            st.session_state.paused_frame = frame.copy()
            st.session_state.current_frame_idx = st.session_state.frame_idx
            st.session_state.fabric_mode = "add"
            st.session_state.current_lost_fish_index = 0
            scroll_to_top()
            cap.release()
            st.rerun()
            break

        # Draw current frame with updated positions
        display_frame = frame.copy()
        for fish_id, bbox in last_positions.items():
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display_frame, fish_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 2)

        # Throttle UI updates
        if st.session_state.frame_idx % 5 == 0:
            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            # Live trajectory update (manual mode only)
            if (st.session_state.get('tracking_source') == 'manual' and st.session_state.get('show_live_trajectory', True)
                and st.session_state.frame_idx - st.session_state.get('last_traj_plot_frame', -1) >= st.session_state.get('traj_refresh_every', 10)):
                traj_placeholder.plotly_chart(_make_traj_fig(st.session_state.trajectories, st.session_state.get('reference_trajectories', {})), use_container_width=True)
                st.session_state.last_traj_plot_frame = st.session_state.frame_idx

        # Update status text
        if lost_in_current_frame:
            status_text.warning(f"⚠️ Lost: {', '.join(lost_in_current_frame)}")
        else:
            status_text.info("✅ Tracking normally")

        st.session_state.frame_idx += 1
        progress_bar.progress(min(st.session_state.frame_idx / frame_count, 1.0))

    cap.release()

# ---------- ENHANCED PAUSE STATE WITH BOUNDING BOXES AND RIGHT-CLICK MENU ----------
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
        cv2.putText(paused_frame_with_rectangles, fish_id, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
    
    # Display status and instructions based on pause type
    if st.session_state.manual_pause:
        st.warning("⏸️ Tracking Manually Paused - Draw NEW rectangles and rename as needed")
    else:
        # Automatic pause - show current lost fish
        current_lost_fish = st.session_state.lost_fish[st.session_state.current_lost_fish_index] if st.session_state.lost_fish else None
        st.error(f"🔴 Tracking Auto-Paused - Draw rectangle for lost fish: **{current_lost_fish}**")
    
    # CREATE SIDEBAR WITH EXISTING FISH IDs AND DOWNLOAD PARTIAL DATA
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
        
        # ---- NEW: DOWNLOAD PARTIAL DATA BUTTON ----
        if len(st.session_state.records) > 0:
            st.divider()
            st.subheader("📥 Download Partial Data")
            partial_df = pd.DataFrame(st.session_state.records,
                                      columns=["frame", "fish_id", "x", "y", "width", "height"])
            partial_df["video_width"] = width
            partial_df["video_height"] = height
            st.download_button(
                label="Download Current Tracking Data (CSV)",
                data=partial_df.to_csv(index=False),
                file_name="partial_tracking_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
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
                            new_tracker = create_tracker(st.session_state.get('tracker_choice', tracker_choice))
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

    # ---- NEW: PARTIAL ANALYSIS WHILE PAUSED ----
    if len(st.session_state.records) > 0:
        with st.expander("📊 Partial Analysis (based on data so far)", expanded=False):
            partial_df = pd.DataFrame(st.session_state.records,
                                      columns=["frame", "fish_id", "x", "y", "width", "height"])
            partial_df["video_width"] = width
            partial_df["video_height"] = height
            
            tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "📏 Size Analysis", "📈 Trajectory"])
            
            with tab1:
                st.markdown("**Fish Location Heatmap (partial)**")
                # Use first frame as background
                heatmap_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                heatmap_canvas = np.zeros((height, width), dtype=np.float32)
                for _, row in partial_df.iterrows():
                    cx, cy = int(row['x']), int(row['y'])
                    if 0 <= cx < width and 0 <= cy < height:
                        heatmap_canvas[cy, cx] += 1
                heatmap_blurred = cv2.GaussianBlur(heatmap_canvas, (51, 51), 0)
                if np.max(heatmap_blurred) > 0:
                    heatmap_norm = (heatmap_blurred / np.max(heatmap_blurred) * 255).astype(np.uint8)
                else:
                    heatmap_norm = heatmap_blurred.astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                alpha = st.slider("Heatmap Opacity (partial)", 0.0, 1.0, 0.6, 0.05, key="partial_alpha")
                blended = cv2.addWeighted(heatmap_background, 1 - alpha, heatmap_colored, alpha, 0)
                st.image(blended, caption="Partial heatmap",use_column_width=True)
                # Download partial heatmap
                heatmap_pil = Image.fromarray(blended)
                buf = BytesIO()
                heatmap_pil.save(buf, format="PNG")
                st.download_button("Download Partial Heatmap PNG", data=buf.getvalue(),
                                   file_name="partial_heatmap.png", mime="image/png")
            
            with tab2:
                st.markdown("**Fish Size Analysis (partial)**")
                size_stats = []
                for fish_id in partial_df['fish_id'].unique():
                    fish_df = partial_df[partial_df['fish_id'] == fish_id]
                    avg_w = fish_df['width'].mean()
                    avg_h = fish_df['height'].mean()
                    avg_area = (avg_w * avg_h) / 1000.0
                    avg_length = max(avg_w, avg_h)
                    size_stats.append({
                        'Fish ID': fish_id,
                        'Avg Width (px)': round(avg_w, 1),
                        'Avg Height (px)': round(avg_h, 1),
                        'Avg Area (kpx²)': round(avg_area, 2),
                        'Est. Length (px)': round(avg_length, 1)
                    })
                size_df_partial = pd.DataFrame(size_stats)
                st.dataframe(size_df_partial, use_container_width=True)
            
            with tab3:
                st.markdown("**Fish Trajectories (partial)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.invert_yaxis()
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.set_title('Partial Fish Movement Trajectories')
                colors = plt.cm.tab20(np.linspace(0, 1, len(partial_df['fish_id'].unique())))
                for i, fish_id in enumerate(partial_df['fish_id'].unique()):
                    fish_df = partial_df[partial_df['fish_id'] == fish_id].sort_values('frame')
                    ax.plot(fish_df['x'], fish_df['y'], marker='.', linestyle='-', markersize=2,
                            color=colors[i], label=fish_id)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                partial_traj_df = partial_df[['frame', 'fish_id', 'x', 'y']].copy()
                partial_traj_df['video_width'] = width
                partial_traj_df['video_height'] = height
                st.download_button(
                    "Download all partial trajectories CSV",
                    data=partial_traj_df.to_csv(index=False),
                    file_name="all_partial_trajectories.csv",
                    mime="text/csv",
                    key="partial_all_traj_dl"
                )
                # Download per-fish partial trajectories
                for fish_id in partial_df['fish_id'].unique():
                    fish_df = partial_df[partial_df['fish_id'] == fish_id][['frame', 'x', 'y']].copy()
                    fish_df['fish_id'] = fish_id
                    fish_df['video_width'] = width
                    fish_df['video_height'] = height
                    csv_fish = fish_df.to_csv(index=False)
                    st.download_button(
                        f"Download {fish_id} partial trajectory CSV",
                        data=csv_fish,
                        file_name=f"{fish_id}_partial_trajectory.csv",
                        mime="text/csv",
                        key=f"partial_dl_{fish_id}"
                    )

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
        st.session_state.trajectories = {}
        st.session_state.last_traj_plot_frame = -1
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
    df["video_width"] = width
    df["video_height"] = height
    csv_path = "multi_fish_tracking.csv"
    df.to_csv(csv_path, index=False)
    st.success(f"✅ Tracking complete! Data saved to {csv_path}")

    # Show raw data preview
    with st.expander("📊 View Raw Tracking Data", expanded=False):
        st.dataframe(df.head(100))
        st.download_button("Download CSV", df.to_csv(index=False), file_name=csv_path, mime="text/csv")

    # ---------- POST-PROCESSING TOOLS (Heatmap & Size Analysis) ----------
    st.subheader("🔬 Post-Processing Analysis")

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "📏 Size Analysis", "📈 Trajectory"])

    with tab1:
        st.markdown("**Fish Location Heatmap**")
        st.info("This heatmap shows the density of fish positions across the video.")

        # Use the first frame as background
        heatmap_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # Create a blank canvas of video dimensions
        heatmap_canvas = np.zeros((height, width), dtype=np.float32)

        # Accumulate positions (center coordinates)
        for _, row in df.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            if 0 <= cx < width and 0 <= cy < height:
                heatmap_canvas[cy, cx] += 1

        # Apply Gaussian blur for smoothness
        heatmap_blurred = cv2.GaussianBlur(heatmap_canvas, (51, 51), 0)

        # Normalize to 0-255 for colormap
        if np.max(heatmap_blurred) > 0:
            heatmap_norm = (heatmap_blurred / np.max(heatmap_blurred) * 255).astype(np.uint8)
        else:
            heatmap_norm = heatmap_blurred.astype(np.uint8)

        # Apply colormap (JET)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend with background
        alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.6, 0.05)
        blended = cv2.addWeighted(heatmap_background, 1 - alpha, heatmap_colored, alpha, 0)

        # Display
        st.image(blended, caption="Heatmap over first frame", use_column_width=True)

        # Option to download heatmap as image
        heatmap_pil = Image.fromarray(blended)
        buf = BytesIO()
        heatmap_pil.save(buf, format="PNG")
        st.download_button("Download Heatmap PNG", data=buf.getvalue(), file_name="fish_heatmap.png", mime="image/png")

    with tab2:
        st.markdown("**Fish Size Analysis**")
        st.info("Bounding box dimensions and estimated length (longer side).")

        # Compute per-fish statistics
        size_stats = []
        for fish_id in df['fish_id'].unique():
            fish_df = df[df['fish_id'] == fish_id]
            avg_w = fish_df['width'].mean()
            avg_h = fish_df['height'].mean()
            avg_area = (avg_w * avg_h) / 1000.0  # in thousands of pixels
            # Estimate length as max of width/height (assuming fish is oriented)
            avg_length = max(avg_w, avg_h)
            size_stats.append({
                'Fish ID': fish_id,
                'Avg Width (px)': round(avg_w, 1),
                'Avg Height (px)': round(avg_h, 1),
                'Avg Area (kpx²)': round(avg_area, 2),
                'Est. Length (px)': round(avg_length, 1)
            })

        size_df = pd.DataFrame(size_stats)
        st.dataframe(size_df, use_container_width=True)

        # Optional: Plot size distribution
        if st.checkbox("Show size distribution plot"):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].bar(size_df['Fish ID'], size_df['Avg Width (px)'], label='Width', alpha=0.7)
            ax[0].bar(size_df['Fish ID'], size_df['Avg Height (px)'], bottom=size_df['Avg Width (px)'], label='Height', alpha=0.7)
            ax[0].set_ylabel('Pixels')
            ax[0].set_title('Average Bounding Box Dimensions')
            ax[0].legend()
            ax[0].tick_params(axis='x', rotation=45)

            ax[1].bar(size_df['Fish ID'], size_df['Est. Length (px)'], color='orange')
            ax[1].set_ylabel('Pixels')
            ax[1].set_title('Estimated Fish Length')
            ax[1].tick_params(axis='x', rotation=45)

            st.pyplot(fig)

    with tab3:
        st.markdown("**Fish Trajectories**")
        st.info("XY coordinates over time for each fish.")

        # Plot trajectories on a blank canvas
        traj_fig, ax = plt.subplots(figsize=(10, 6))
        # Invert Y axis because image origin is top-left
        ax.invert_yaxis()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Fish Movement Trajectories')

        colors = plt.cm.tab20(np.linspace(0, 1, len(df['fish_id'].unique())))
        for i, fish_id in enumerate(df['fish_id'].unique()):
            fish_df = df[df['fish_id'] == fish_id].sort_values('frame')
            ax.plot(fish_df['x'], fish_df['y'], marker='.', linestyle='-', markersize=2, color=colors[i], label=fish_id)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(traj_fig)

        all_traj_df = df[['frame', 'fish_id', 'x', 'y']].copy()
        all_traj_df['video_width'] = width
        all_traj_df['video_height'] = height
        st.download_button(
            "Download all trajectories CSV",
            data=all_traj_df.to_csv(index=False),
            file_name="all_trajectories.csv",
            mime="text/csv",
            key="all_traj_dl"
        )

        # Option to download trajectory data per fish
        for fish_id in df['fish_id'].unique():
            fish_df = df[df['fish_id'] == fish_id][['frame', 'x', 'y']].copy()
            fish_df['fish_id'] = fish_id
            fish_df['video_width'] = width
            fish_df['video_height'] = height
            csv_fish = fish_df.to_csv(index=False)
            st.download_button(
                f"Download {fish_id} trajectory CSV",
                data=csv_fish,
                file_name=f"{fish_id}_trajectory.csv",
                mime="text/csv",
                key=f"dl_{fish_id}"
            )

    # Reset tracking state button
    if st.button("Start New Tracking Session", use_container_width=True):
        st.session_state.tracking_started = False
        st.session_state.tracking_paused = False
        st.session_state.manual_pause = False
        st.session_state.trackers = []
        st.session_state.records = []
        st.session_state.trajectories = {}
        st.session_state.last_traj_plot_frame = -1
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