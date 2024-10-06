import mujoco
import mujoco.viewer
import time
import os
import xml.etree.ElementTree as ET
import shutil
import threading
import numpy as np
import atexit
import signal
import sys

global_viewer = None
stop_event = threading.Event()

STRUCTURE_COLLAPSED = False
initial_positions = {}

ORIGINAL_XML_PATH = "portal_frame_original.xml"
#ORIGINAL_XML_PATH = "mjmodel.xml"
#WORKING_XML_PATH = "portal_frame.xml"
WORKING_XML_PATH = "mjmodel.xml"

def fix_element_in_simulation(xml_path, element_name):
    """Fix the specified element in the simulation by removing its freejoint."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    for element in worldbody.findall('body'):
        if element.get('name') == element_name:
            joint = element.find('joint')
            if joint is not None:
                element.remove(joint)  # Remove the freejoint to fix the element
                print(f"Fixed element: {element_name}")
                tree.write(xml_path)
                return True
            else:
                print(f"No freejoint found for element: {element_name}")
    print(f"Element {element_name} not found in XML.")
    return False

def re_add_element(xml_path, original_xml_path, element_name):
    """Re-add the removed element from the original XML."""
    original_tree = ET.parse(original_xml_path)
    original_root = original_tree.getroot()
    original_worldbody = original_root.find('worldbody')
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    for element in original_worldbody.findall('body'):
        if element.get('name') == element_name:
            worldbody.append(element)  # Re-add the element
            print(f"Re-added element: {element_name}")
            tree.write(xml_path)
            return

def ensure_free_joints(xml_path):
    """Ensure all bodies in the XML have a free joint."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    for element in worldbody.findall('body'):
        joint = element.find('joint')
        if joint is None:
            # Add a free joint if it doesn't exist
            ET.SubElement(element, 'joint', type='free', limited='false', actuatorfrclimited='false')
            print(f"Added free joint to body: {element.get('name')}")
    
    # Write changes back to the XML file
    tree.write(xml_path)

def get_bounding_box(pos, size):
    """Calculate the bounding box for a body given its position and size."""
    x, y, z = pos
    sx, sy, sz = size
    return (x - sx/2, x + sx/2, y - sy/2, y + sy/2, z - sz/2, z + sz/2)

def are_boxes_attached(box1, box2, threshold=0.001):
    """Determine if two bounding boxes are attached based on a proximity threshold."""
    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = box1
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = box2


    # Check for overlap or proximity in each dimension
    x_attached = (x1_min <= x2_max + threshold and x1_max >= x2_min - threshold)
    y_attached = (y1_min <= y2_max + threshold and y1_max >= y2_min - threshold)
    z_attached = (z1_min <= z2_max + threshold and z1_max >= z2_min - threshold)

    return x_attached and y_attached and z_attached

def find_attached_elements(selected_element, model):
    """Find elements attached to the selected element."""
    attached_elements = []
    try:
        selected_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, selected_element)
    except Exception as e:
        print(f"Error finding body ID for {selected_element}: {e}")
        return attached_elements

    selected_pos = model.body_pos[selected_body]
    selected_size = model.geom_size[selected_body]
    selected_box = get_bounding_box(selected_pos, selected_size)

    for i in range(model.nbody):
        if i == selected_body:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_pos = model.body_pos[i]
        body_size = model.geom_size[i]
        body_box = get_bounding_box(body_pos, body_size)

        if are_boxes_attached(selected_box, body_box):
            attached_elements.append(body_name)

    return attached_elements

def force_exit(signum, frame):
    print("\nForce exiting...")
    os._exit(1)

def signal_handler(signum, frame):
    print("\nReceived interrupt, cleaning up...")
    cleanup()
    print("Exiting...")
    
    # Set a timer to force exit if cleanup takes too long
    threading.Timer(5.0, force_exit, args=(signum, frame)).start()

def cleanup():
    print("Cleaning up...")
    close_viewer()
    reset_xml()

def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def close_viewer():
    global global_viewer
    stop_event.set()
    if 'viewer_thread' in globals() and viewer_thread and viewer_thread.is_alive():
        viewer_thread.join(timeout=5)
    if global_viewer and global_viewer.is_running():
        global_viewer.close()
    global_viewer = None
    stop_event.clear()

def remove_element(xml_path, element_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    for element in worldbody.findall('body'):
        if element.get('name') == element_name:
            worldbody.remove(element)
            tree.write(xml_path)
            print(f"Removed element: {element_name}")
            return True
    
    print(f"Element {element_name} not found")
    return False

def print_model_info(model):
    print(f"Number of bodies: {model.nbody}")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body {i}: {body_name}")

def initialize_positions(model, data):
    global initial_positions
    initial_positions = {}
    for i in range(0, model.nbody-1):
        if 7*i+3 <= len(data.qpos):
            initial_positions[i] = data.qpos[7*i:7*i+3].copy()
            print (initial_positions[i])
        else:
            #print(f"Warning: Unable to initialize position for body {i}")
            initial_positions[i] = model.body_pos[i].copy()
            print (initial_positions[i])

def is_structure_collapsed(model, data, displacement_threshold=0.5, rotation_threshold=0.5, floor_threshold=0.1):
    for i in range(0, model.nbody -1 ):
        # Check if the body still exists in the model
        if 7*i+7 > len(data.qpos):
            continue  # Skip this body if it no longer exists

        # Check total displacement
        current_pos = data.qpos[7*i:7*i+3]
        if i not in initial_positions:
            print(f"Warning: No initial position for body {i}")
            continue
        initial_pos = initial_positions[i]
        displacement = np.linalg.norm(current_pos - initial_pos)
        
        # Check rotation (simplified, using quaternion)
        quat = data.qpos[7*i+3:7*i+7]
        if len(quat) > 0:
            rotation_angle = 2 * np.arccos(np.abs(quat[0]))  # Angle from identity rotation
        else:
            rotation_angle = 0  # Default to no rotation if quaternion is empty
        
        # Check if body is close to the floor
        height_above_floor = current_pos[2]  # z-coordinate
        
        print(f"Debug: Body {i} - Displacement: {displacement:.4f}, Rotation: {rotation_angle:.4f}, Height: {height_above_floor:.4f}")
        
        if displacement > displacement_threshold or rotation_angle > rotation_threshold or height_above_floor < floor_threshold:
            return True
    return False

def reset_xml():
    shutil.copy2(ORIGINAL_XML_PATH, WORKING_XML_PATH)
    print("Reset XML file to original state")
    destination_path = "mjmodel.xml"
    shutil.copy2(ORIGINAL_XML_PATH, destination_path)

def run_simulation():
    global STRUCTURE_COLLAPSED, global_viewer, viewer_thread
    # Ensure all elements have free joints before starting the simulation
    

    def run_viewer(model, data):
        global global_viewer
        try:
            global_viewer = mujoco.viewer.launch(model, data)
            if global_viewer is None:
                print("Failed to launch MuJoCo viewer. Simulation will run without visualization.")
                return
            
            def align_view():
                global_viewer.cam.lookat[:] = model.stat.center
                global_viewer.cam.distance = 1 * model.stat.extent  # Increase this value to zoom out
                global_viewer.cam.azimuth = 90
                global_viewer.cam.elevation = -40

            align_view()  # Initial alignment
            
            while not stop_event.is_set() and global_viewer.is_running():
                global_viewer.sync()
                time.sleep(0.01)
        except Exception as e:
            print(f"An error occurred in the viewer thread: {e}")
        finally:
            if global_viewer and global_viewer.is_running():
                global_viewer.close()
            global_viewer = None

    try:
        while True:
            STRUCTURE_COLLAPSED = False
            model, data = load_model(WORKING_XML_PATH)
            print("\nCurrent model information:")
            print_model_info(model)
            initialize_positions(model, data)

            close_viewer()  # Ensure previous viewer is closed

            viewer_thread = threading.Thread(target=run_viewer, args=(model, data))
            viewer_thread.start()

            user_input = input("\nEnter element to remove (column1, column2, beam) or 'q' to quit: ")

            if user_input.lower() == 'q':
                break

            if user_input in ["column1", "column2", "beam"]:
                ensure_free_joints(WORKING_XML_PATH)
                if remove_element(WORKING_XML_PATH, user_input):
                    print(f"Element {user_input} removed. Restarting simulation...")
                    
                    model, data = load_model(WORKING_XML_PATH)
                    initialize_positions(model, data)
                    
                    for _ in range(500):
                        mujoco.mj_step(model, data)
                        if is_structure_collapsed(model, data):
                            STRUCTURE_COLLAPSED = True
                            print("Log: collapsed: true")
                            attached_elements = find_attached_elements(user_input, model)
                            print(f"Attached elements: {attached_elements}")
                            
                            
                            if len(attached_elements) == 1:                               
                                attached_element = attached_elements[0]
                                if fix_element_in_simulation(WORKING_XML_PATH, attached_element):
                                    print(f"Element {attached_element} is now fixed.")
                                
                                #if re_add_element(WORKING_XML_PATH, ORIGINAL_XML_PATH, user_input):
                                    #print(f"Element {user_input} re-added to the simulation.")
                               # Reload the model and re-run the simulation steps
                                model, data = load_model(WORKING_XML_PATH)
                                initialize_positions(model, data)
                                STRUCTURE_COLLAPSED = False
                                for _ in range(500):
                                   mujoco.mj_step(model, data)
                                   if is_structure_collapsed(model, data):
                                    break
                                time.sleep(0.01)    
                                    
                                if not STRUCTURE_COLLAPSED:
                                    print(f"Log: Supporting: {attached_element}")
                                    print(f"supporting {attached_element} allows removing {user_input}")
                                #ensure_free_joints(WORKING_XML_PATH)


                            break
                        time.sleep(0.01)
                    
                    if not STRUCTURE_COLLAPSED:
                        print("Log: collapsed: false")
                else:
                    print("Failed to remove element. Continuing with current model.")
            else:
                print("Invalid input. Please try again.")

            print(f"Debug: STRUCTURE_COLLAPSED = {STRUCTURE_COLLAPSED}")

            close_viewer()  # Close viewer after each simulation step

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received in run_simulation")
    finally:
        print("Exiting run_simulation")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Set up the signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_simulation()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Final cleanup")
        cleanup()

    print("Program finished")
    sys.exit(0)
