import argparse, sys, os, math, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['View Layer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['View Layer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['View Layer'].use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera(target_location=None):
    """Create a camera. We do NOT add a TRACK_TO constraint here because
    Blender's TRACK_TO always uses world +Z as the up-reference, which forces
    the rendered image's up to be world +Z. We instead build the camera's
    matrix_world manually per-view (see main loop) so we can choose world +Y
    as the image-up reference.
    """
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    return cam

def disable_object_shadows():
    """Disables shadow visibility for all mesh objects in the scene."""
    print('[INFO] Disabling object self-shadows...')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Eevee / Workbench
            obj.visible_shadow = False
            # Cycles
            if hasattr(obj, 'cycles_visibility'):
                obj.cycles_visibility.shadow = False

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # 1. Setup World Background (Environment Light)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    # Ensure we have a background node
    if 'Background' in world.node_tree.nodes:
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (1, 1, 1, 1) # White Color
        bg.inputs[1].default_value = 1.0 # Strength
    
    # Helper to disable shadows on lights
    def configure_no_shadow(light_obj):
        light_obj.data.use_shadow = False
        if hasattr(light_obj.data, 'cycles'):
            light_obj.data.cycles.cast_shadow = False

    # 2. Create Scene Lights (No Shadows)
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    configure_no_shadow(default_light)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    configure_no_shadow(top_light)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    configure_no_shadow(bottom_light)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    elif file_extension == "obj":
        import_function(filepath=object_path, axis_forward='Y', axis_up='Z')
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                obj.matrix_world.identity()
    else:
        import_function(filepath=object_path)
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat


def _build_color_sampler(material):
    """Return (pixels[H,W,3] or None, fallback_rgb[3]) for a material."""
    fallback = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if material is None or not material.use_nodes:
        return None, fallback
    color_socket = None
    for n in material.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            color_socket = n.inputs['Base Color']
            break
        if n.type == 'BSDF_DIFFUSE':
            color_socket = n.inputs['Color']
            break
    if color_socket is None:
        return None, fallback
    if color_socket.is_linked:
        src = color_socket.links[0].from_node
        if src.type == 'TEX_IMAGE' and src.image is not None:
            img = src.image
            w, h = img.size
            ch = img.channels
            if w > 0 and h > 0 and ch >= 3:
                pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)[:, :, :3]
                return pixels, fallback
    return None, np.array(color_socket.default_value[:3], dtype=np.float32)


def _sample_texture(pixels, fallback, uvs):
    if pixels is None:
        return np.tile(fallback[None, :], (len(uvs), 1))
    h, w, _ = pixels.shape
    u = np.mod(uvs[:, 0], 1.0)
    v = np.mod(uvs[:, 1], 1.0)
    x = np.clip((u * w).astype(np.int32), 0, w - 1)
    y = np.clip((v * h).astype(np.int32), 0, h - 1)
    return pixels[y, x]


def sample_mesh_surface(num_points=10000, seed=0):
    """Area-weighted surface sampling with UV-based color sampling."""
    deps = bpy.context.evaluated_depsgraph_get()
    v0_list, v1_list, v2_list = [], [], []
    uv0_list, uv1_list, uv2_list = [], [], []
    mat_idx_list = []
    samplers = []

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            uv_data = mesh.uv_layers.active.data if mesh.uv_layers.active else None

            base = len(samplers)
            slots = list(obj.material_slots)
            if slots:
                for slot in slots:
                    samplers.append(_build_color_sampler(slot.material))
            else:
                samplers.append(_build_color_sampler(None))

            wm = np.array(obj.matrix_world, dtype=np.float32)
            verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
            ones = np.ones((len(verts), 1), dtype=np.float32)
            verts_w = (np.concatenate([verts, ones], axis=1) @ wm.T)[:, :3]

            for tri in mesh.loop_triangles:
                vi = tri.vertices
                li = tri.loops
                v0_list.append(verts_w[vi[0]]); v1_list.append(verts_w[vi[1]]); v2_list.append(verts_w[vi[2]])
                if uv_data is not None:
                    uv0_list.append(uv_data[li[0]].uv[:])
                    uv1_list.append(uv_data[li[1]].uv[:])
                    uv2_list.append(uv_data[li[2]].uv[:])
                else:
                    uv0_list.append((0.0, 0.0)); uv1_list.append((0.0, 0.0)); uv2_list.append((0.0, 0.0))
                m = tri.material_index if (slots and tri.material_index < len(slots)) else 0
                mat_idx_list.append(base + m)
        finally:
            eval_obj.to_mesh_clear()

    if not v0_list:
        raise RuntimeError("No mesh triangles to sample.")

    v0 = np.asarray(v0_list, dtype=np.float32)
    v1 = np.asarray(v1_list, dtype=np.float32)
    v2 = np.asarray(v2_list, dtype=np.float32)
    uv0 = np.asarray(uv0_list, dtype=np.float32)
    uv1 = np.asarray(uv1_list, dtype=np.float32)
    uv2 = np.asarray(uv2_list, dtype=np.float32)
    mat_idx = np.asarray(mat_idx_list, dtype=np.int32)

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise RuntimeError("Mesh has zero total surface area.")
    probs = areas / total

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(areas), size=num_points, p=probs)
    r1 = rng.random(num_points).astype(np.float32)
    r2 = rng.random(num_points).astype(np.float32)
    s = np.sqrt(r1)
    b0 = (1.0 - s)[:, None]
    b1 = (s * (1.0 - r2))[:, None]
    b2 = (s * r2)[:, None]

    points = b0 * v0[sel] + b1 * v1[sel] + b2 * v2[sel]
    uvs = b0 * uv0[sel] + b1 * uv1[sel] + b2 * uv2[sel]

    colors = np.empty((num_points, 3), dtype=np.float32)
    sel_mat = mat_idx[sel]
    for mi in np.unique(sel_mat):
        mask = sel_mat == mi
        pixels, fallback = samplers[mi]
        colors[mask] = _sample_texture(pixels, fallback, uvs[mask])

    colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return points.astype(np.float32), colors


def write_points_ply(path, points, colors):
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                             ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'] = points[:, 0]; rec['y'] = points[:, 1]; rec['z'] = points[:, 2]
    rec['r'] = colors[:, 0]; rec['g'] = colors[:, 1]; rec['b'] = colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(rec.tobytes())


def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
    print('[INFO] Scene initialized.')
    
    bbox_min_orig, bbox_max_orig = scene_bbox()
    scene_center = (bbox_min_orig + bbox_max_orig) / 2
    scene_extent = max(bbox_max_orig - bbox_min_orig)
    if scene_extent <= 1e-6:
        scene_extent = 1.0

    if arg.normalize:
        scale, offset = normalize_scene()
        print('[INFO] Scene normalized (scale applied).')
        bbox_min_norm, bbox_max_norm = scene_bbox()
        camera_center = (bbox_min_norm + bbox_max_norm) / 2
        camera_extent = max(bbox_max_norm - bbox_min_norm)
    else:
        scale = 1.0
        offset = Vector((0.0, 0.0, 0.0))
        camera_center = scene_center
        camera_extent = scene_extent

    if camera_extent <= 1e-6:
        camera_extent = 1.0
    
    # Initialize camera and lighting
    cam = init_camera(target_location=camera_center)
    init_lighting()
    # DISABLE OBJECT SHADOWS
    disable_object_shadows()
    print('[INFO] Camera and lighting initialized (Shadows Disabled).')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[float(bbox_min_orig.x), float(bbox_min_orig.y), float(bbox_min_orig.z)],
                 [float(bbox_max_orig.x), float(bbox_max_orig.y), float(bbox_max_orig.z)]],
        "scale": scale,
        "offset": [float(offset.x), float(offset.y), float(offset.z)],
        "scene_center": [float(camera_center.x), float(camera_center.y), float(camera_center.z)],
        "frames": []
    }
    views = json.loads(arg.views)
    world_up = Vector((0.0, 1.0, 0.0))
    for i, view in enumerate(views):
        radius = view['radius'] * camera_extent
        yaw = float(view['yaw'])
        pitch = float(view['pitch'])

        # Y-up spherical placement:
        #   c = center + r * (cos(yaw)*cos(pitch), sin(pitch), sin(yaw)*cos(pitch))
        # so pitch is elevation about world Y, yaw is azimuth in the X-Z plane
        # measured from +X toward +Z (CCW when viewed from +Y).
        cam_pos = Vector((
            camera_center.x + radius * np.cos(yaw) * np.cos(pitch),
            camera_center.y + radius * np.sin(pitch),
            camera_center.z + radius * np.sin(yaw) * np.cos(pitch),
        ))

        # Aim camera at scene center with world +Y as image-up.
        forward = (camera_center - cam_pos).normalized()
        right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()
        # Blender camera looks down its local -Z, so local +Z = -forward.
        cam.matrix_world = Matrix((
            (right.x, up.x, -forward.x, cam_pos.x),
            (right.y, up.y, -forward.y, cam_pos.y),
            (right.z, up.z, -forward.z, cam_pos.z),
            (0.0,     0.0,  0.0,        1.0),
        ))
        bpy.context.view_layer.update()

        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_points:
        points, colors = sample_mesh_surface(num_points=arg.num_points, seed=arg.seed)
        write_points_ply(os.path.join(arg.output_folder, 'mesh.ply'), points, colors)
        print(f'[INFO] Saved {len(points)} surface-sampled points with UV-textured colors.')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_points', action='store_true', help='Area-weighted surface sample points with UV colors into mesh.ply.')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of surface-sampled points for 3DGS init.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for surface sampling.')
    parser.add_argument('--normalize', action='store_true', help='Scale and recenter the mesh to unit cube before rendering.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)