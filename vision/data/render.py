import os
import json
import numpy as np
import subprocess
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_OBJ_DIR = os.path.join(REPO_ROOT, "data", "ObjectFolder-Real", "meshes", "full")
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "vision", "data", "outputs")
BLENDER_SCRIPT_PATH = os.path.join(REPO_ROOT, "vision", "data", "blender.py")

# Blender Download
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALL_DIR = os.path.join(REPO_ROOT, "blender")
BLENDER_EXEC_PATH = os.path.join(BLENDER_INSTALL_DIR, 'blender-3.0.1-linux-x64', 'blender')


def install_blender():

    if os.path.exists(BLENDER_EXEC_PATH):
        print(f"Blender executable found at: {BLENDER_EXEC_PATH}")
        return BLENDER_EXEC_PATH

    print(f"Blender not found at {BLENDER_EXEC_PATH}. Attempting to install...")

    os.makedirs(BLENDER_INSTALL_DIR, exist_ok=True)
    tarball_path = os.path.join(BLENDER_INSTALL_DIR, 'blender-3.0.1-linux-x64.tar.xz')
    if not os.path.exists(tarball_path):
        print(f"Downloading Blender from {BLENDER_LINK}...")
        wget_cmd = ['wget', BLENDER_LINK, '-P', BLENDER_INSTALL_DIR]
        try:
            subprocess.check_call(wget_cmd)
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            return None
        except FileNotFoundError:
            print("Error: 'wget' command not found. Please install wget or download Blender manually.")
            return None
    else:
        print(f"Blender tarball already exists at: {tarball_path}")

    print(f"Extracting Blender to {BLENDER_INSTALL_DIR}...")
    tar_cmd = ['tar', '-xvf', tarball_path, '-C', BLENDER_INSTALL_DIR]
    try:
        subprocess.check_call(tar_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")
        return None
    except FileNotFoundError:
        print("Error: 'tar' command not found.")
        return None

    if os.path.exists(BLENDER_EXEC_PATH):
        print(f"Blender successfully installed at: {BLENDER_EXEC_PATH}")
        return BLENDER_EXEC_PATH
    else:
        print(f"Installation failed. Executable not found at expected path: {BLENDER_EXEC_PATH}")
        return None

def generate_sonicgauss_views(num_azimuth=24, elevations=[-45, 0, 45], radius=2, fov_deg=50):
    """
    Y-up orbit, matched to the (Y-up) camera placement in blender.py:
        cam = center + r * (cos(yaw)*cos(pitch), sin(pitch), sin(yaw)*cos(pitch))
    blender.py also orients the camera so the rendered image-up is world +Y.

    Three rings at elevations -45, 0, +45 deg about world Y. Each ring has
    `num_azimuth` steps starting at yaw=0 (camera on +X side) and rotating
    counter-clockwise when viewed from +Y, sweeping +X -> +Z -> -X -> -Z.

    With these defaults, frame 000 has the camera on the +X side at -45 deg
    elevation, and the world +Z axis appears on the LEFT of the image.
    """
    views = []
    az_step = 360 / num_azimuth
    for el_deg in elevations:
        pitch = float(np.radians(el_deg))
        for i in range(num_azimuth):
            yaw = float(np.radians(i * az_step))
            views.append({
                'yaw': yaw,
                'pitch': pitch,
                'radius': radius,
                'fov': np.radians(fov_deg),
            })
    return views

def main(obj_path, output_dir):

    BLENDER_PATH = install_blender()
    
    if BLENDER_PATH is None:
        print("Could not install or find Blender. Exiting.")
        return

    if not os.path.exists(obj_path):
        print(f"Error: Input file not found at {obj_path}")
        return
        
    if not os.path.exists(BLENDER_SCRIPT_PATH):
        print(f"Error: Blender script not found at {BLENDER_SCRIPT_PATH}")
        return

    os.makedirs(output_dir, exist_ok=True)

    views = generate_sonicgauss_views()

    # run blender in background mode with the specified script and parameters
    cmd = [
        BLENDER_PATH,
        '-b',
        '-P', BLENDER_SCRIPT_PATH,
        '--',
        '--views', json.dumps(views),
        '--object', obj_path,
        '--output_folder', output_dir,
        '--resolution', '800',
        '--engine', 'CYCLES',
        '--save_points',
        '--num_points', '10000',
    ]

    print("Running Blender command:")
    print(" ".join(cmd))

    try:
        # run the command
        subprocess.check_call(cmd)
        print(f"\nSuccessfully rendered images and transforms.json to {output_dir}")
        
        json_path = os.path.join(output_dir, 'transforms.json')
        if os.path.exists(json_path):
            print(f"Confirmed output: {json_path}")
        else:
            print("Warning: transforms.json was not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"Blender execution failed with error: {e}")
    except FileNotFoundError:
        print(f"Error: Blender executable '{BLENDER_PATH}' not found.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Render a specific object using Blender.")
    parser.add_argument('object_id', type=str, 
                        help='The ID of the object to render (e.g., "2", "98").')
    args = parser.parse_args()

    obj_path = os.path.join(BASE_OBJ_DIR, args.object_id, "model.obj")
    output_dir = os.path.join(BASE_OUTPUT_DIR, args.object_id)
    
    print(f"--- Processing Object ID: {args.object_id} ---")
    print(f"Input Path: {obj_path}")
    print(f"Output Path: {output_dir}")
    print("---------------------------------")

    main(obj_path, output_dir)