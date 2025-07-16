import os
import trimesh
import numpy as np
#from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Animation, AnimationChannel, AnimationSampler, BufferTarget, FLOAT, UNSIGNED_SHORT, Position, Weights, MorphTarget, MeshPrimitive, PERSPECTIVE
from pygltflib import * 
import argparse
import glob
import shutil

# Helper function to find unintentional type objects
def find_type_objects(obj, path="root"):
    if isinstance(obj, type):
        raise TypeError(f"⚠️ Type object found at: {path} -> {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_type_objects(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_type_objects(item, f"{path}[{i}]")
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            find_type_objects(v, f"{path}.{k}")

# === Load texture and normal maps ===
def load_image_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def main(input_root, frame_dir, output_glb, FPS):
    # === Input Properties ===
    framerate = FPS
    
    # === Load .obj sequence ===
    os.makedirs(frame_dir, exist_ok=True)

    # === Gather all .obj files from subfolders ===
    all_obj_paths = glob.glob(os.path.join(input_root, "**", "*.obj"), recursive=True)
    obj_paths = sorted([p for p in all_obj_paths if not p.endswith("_detail.obj")])

    # Copy them into frame_dir
    for src_path in obj_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(frame_dir, filename)
        shutil.copy2(src_path, dst_path)

    # === Get the folder of the first frame ===
    first_frame_folder = os.path.dirname(obj_paths[0])
    base_name = os.path.basename(obj_paths[0]).replace(".obj", "")

    # === Expected texture and normal map filenames ===
    texture_path = os.path.join(first_frame_folder, f"{base_name}.png")
    normal_map_path = os.path.join(first_frame_folder, f"{base_name}_normals.png")

    # === Copy texture and normal map ===
    for file_path in [texture_path, normal_map_path]:
        if os.path.isfile(file_path):
            shutil.copy2(file_path, os.path.join(frame_dir, os.path.basename(file_path)))
            print(f"✅ Copied: {file_path}")
        else:
            print(f"⚠️ Missing file: {file_path}")
    
    # === Continue with GLB conversion ===
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])
    has_animation = len(frame_paths) > 1
    if not has_animation:
        print("ℹ️ Only one frame detected - exporting static model")

    # Load base mesh
    base_mesh = trimesh.load(frame_paths[0], process=False, maintain_order=True)
    base_vertices = base_mesh.vertices
    base_normals = base_mesh.vertex_normals
    base_uvs = base_mesh.visual.uv
    base_uvs[:, 1] = 1.0 - base_uvs[:, 1]  # Flip V coordinate
    faces = base_mesh.faces

    morph_deltas = []
    if has_animation:
        for i, frame in enumerate(frame_paths[1:]):
            mesh = trimesh.load(frame, process=False, maintain_order=True)
            if mesh.vertices.shape != base_vertices.shape:
                raise ValueError(
                    f"Frame {i+1} vertex count mismatch: "
                    f"{mesh.vertices.shape[0]} vs {base_vertices.shape[0]}"
                )
            if not np.array_equal(mesh.faces, faces):
                raise ValueError(f"Topology mismatch in frame {i+1}")
            delta = mesh.vertices - base_vertices
            morph_deltas.append(delta.astype(np.float32))

        morph_deltas = np.array(morph_deltas)
        frame_count = morph_deltas.shape[0] + 1

    # === Set up GLTF2 ===
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.animations = []

    # Initialize buffers
    gltf.buffers = [Buffer(byteLength=0)]
    gltf.bufferViews = []
    gltf.accessors = []
    gltf.meshes = []
    gltf.nodes = []
    gltf.images = []
    gltf.textures = []
    gltf.materials = []

    buffer_data = bytearray()
    offset = 0

    # === Helper Function for Adding Buffer Data ===
    def add_buffer(data, target=None):
        nonlocal offset, buffer_data
        # Handle both numpy arrays and bytes
        if hasattr(data, 'tobytes'):  # For numpy arrays
            bytes_data = data.tobytes()
        else:  # For raw bytes
            bytes_data = data
        
        # Align to 4 bytes
        padding = (4 - (offset % 4)) % 4
        if padding > 0:
            buffer_data += b'\x00' * padding
            offset += padding
            
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(bytes_data),
            target=target
        ))
        
        buffer_data += bytes_data
        offset += len(bytes_data)
        
        return len(gltf.bufferViews) - 1

    # === Add Geometry Data ===
    # Positions
    bv_pos = add_buffer(base_vertices.astype(np.float32), ARRAY_BUFFER)
    a_pos = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_pos,
        componentType=FLOAT,
        count=len(base_vertices),
        type="VEC3"
    ))
    
    # Normals
    bv_norm = add_buffer(base_normals.astype(np.float32), ARRAY_BUFFER)
    a_norm = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_norm,
        componentType=FLOAT,
        count=len(base_normals),
        type="VEC3"
    ))
    
    # UVs
    bv_uv = add_buffer(base_uvs.astype(np.float32), ARRAY_BUFFER)
    a_uv = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_uv,
        componentType=FLOAT,
        count=len(base_uvs),
        type="VEC2"
    ))
    
    # Faces
    bv_faces = add_buffer(faces.flatten().astype(np.uint16), ELEMENT_ARRAY_BUFFER)
    a_faces = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_faces,
        componentType=UNSIGNED_SHORT,
        count=len(faces) * 3,
        type="SCALAR"
    ))

    # === Add Morph Targets ===
    morph_target_accessors = []
    for delta in morph_deltas:
        bv_delta = add_buffer(delta, ARRAY_BUFFER)
        a_delta = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_delta,
            componentType=FLOAT,
            count=len(delta),
            type="VEC3",
            min=delta.min(axis=0).tolist(),
            max=delta.max(axis=0).tolist()
        ))
        morph_target_accessors.append(a_delta)

    # === Add Textures ===
    image_paths = {
        "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
        "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
    }

    image_indices = {}
    for key, path in image_paths.items():
        if os.path.isfile(path):
            img_data = load_image_bytes(path)
            bv_img = add_buffer(img_data)
            
            image = Image(bufferView=bv_img, mimeType="image/png")
            gltf.images.append(image)
            
            texture = Texture(source=len(gltf.images)-1)
            gltf.textures.append(texture)
            
            image_indices[key] = len(gltf.textures)-1
        else:
            print(f"⚠️ Image missing: {path}")

    # === Create Material ===
    pbr = PbrMetallicRoughness()
    if "baseColor" in image_indices:
        pbr.baseColorTexture = TextureInfo(index=image_indices["baseColor"])

    material = Material(
        pbrMetallicRoughness=pbr,
        normalTexture=TextureInfo(index=image_indices["normalMap"]) if "normalMap" in image_indices else None
    )
    gltf.materials.append(material)

    # === Create Mesh ===
    primitive = Primitive(
        attributes={
            "POSITION": a_pos,
            "NORMAL": a_norm,
            "TEXCOORD_0": a_uv
        },
        indices=a_faces,
        targets=[{"POSITION": a} for a in morph_target_accessors] if has_animation else None,
        material=0
    )

    mesh = Mesh(
        primitives=[primitive],
        weights=[0.0] * len(morph_target_accessors) if has_animation else None
    )
    gltf.meshes.append(mesh)
    
    node = Node(mesh=0)
    gltf.nodes.append(node)

    # === Create Animation ===
    if has_animation and morph_target_accessors:
        # Time data
        time_data = np.arange(1, frame_count) * (1.0 / framerate)
        bv_time = add_buffer(time_data.astype(np.float32))
        a_time = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_time,
            componentType=FLOAT,
            count=len(time_data),
            type="SCALAR",
            min=[float(time_data[0])],
            max=[float(time_data[-1])]
        ))

        # Weight data
        weight_data = np.zeros((len(time_data), len(morph_target_accessors)), dtype=np.float32)
        for i in range(len(time_data)):
            weight_data[i, i] = 1.0  # Only one active morph per frame
        
        bv_weights = add_buffer(weight_data)
        a_weights = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_weights,
            componentType=FLOAT,
            count=weight_data.size,
            type="SCALAR"
        ))

        # Animation setup
        sampler = AnimationSampler(
            input=a_time,
            output=a_weights,
            interpolation="STEP"
        )
        channel = AnimationChannel(
            sampler=len(gltf.animations[0].samplers) if gltf.animations else 0,
            target={"node": 0, "path": "weights"}
        )
        animation = Animation(samplers=[sampler], channels=[channel])
        gltf.animations.append(animation)

    # === Finalize buffer ===
    gltf.buffers[0].byteLength = len(buffer_data)
    gltf.set_binary_blob(buffer_data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_glb), exist_ok=True)
    gltf.save_binary(output_glb)
    print(f"✅ Exported to {output_glb}")

#def main(input_root, frame_dir, output_glb, FPS):
    # # === Input Properties ===
    # framerate = FPS
    
    # # === Load .obj sequence ===
    # input_root = input_root
    # frame_dir = frame_dir
    # os.makedirs(frame_dir, exist_ok=True)

    # # === Gather all .obj files from subfolders ===
    # all_obj_paths = glob.glob(os.path.join(input_root, "**", "*.obj"), recursive=True)
    # obj_paths = sorted([p for p in all_obj_paths if not p.endswith("_detail.obj")])

    # # Optionally: Copy them into frame_dir for clean naming
    # for src_path in obj_paths:
    #     filename = os.path.basename(src_path)
    #     dst_path = os.path.join(frame_dir, filename)
    #     shutil.copy2(src_path, dst_path)

    # # === Get the folder of the first frame ===
    # first_frame_folder = os.path.dirname(obj_paths[0])
    # base_name = os.path.basename(obj_paths[0]).replace(".obj", "")

    # # === Expected texture and normal map filenames ===
    # texture_path = os.path.join(first_frame_folder, f"{base_name}.png")
    # normal_map_path = os.path.join(first_frame_folder, f"{base_name}_normals.png")

    # # === Copy texture and normal map if they exist ===
    # for file_path in [texture_path, normal_map_path]:
    #     if os.path.isfile(file_path):
    #         shutil.copy2(file_path, os.path.join(frame_dir, os.path.basename(file_path)))
    #         print(f"✅ Copied: {file_path}")
    #     else:
    #         print(f"⚠️ Missing file: {file_path}")
    
    # # === Continue with your original GLB conversion logic using frame_dir ===
    # frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])
    # has_animation = len(frame_paths) > 1
    # #if len(frame_paths) < 2:
    #     #raise RuntimeError("Need at least 2 frames for morph animation")
    # if not has_animation:
    #     print("ℹ️ Only one frame detected — exporting static model without morph animation.")

    # # Load base mesh and others
    # base_mesh = trimesh.load(frame_paths[0], process=False, maintain_order=True)
    # print("Has visual:", base_mesh.visual.kind)  # should say 'texture'
    # print("UV shape:", base_mesh.visual.uv.shape)
    # base_vertices = base_mesh.vertices
    # base_normals = base_mesh.vertex_normals  # Vertex normals
    # base_uvs = base_mesh.visual.uv  # UV coordinates
    # base_uvs[:,1] = 1.0 - base_uvs[:,1]
    # faces = base_mesh.faces

    # morph_deltas = []
    # if has_animation :
    #     for i, frame in enumerate(frame_paths[1:]):
    #         mesh = trimesh.load(frame, process=False, maintain_order=True)
    #         if mesh.vertices.shape != base_vertices.shape:
    #             raise ValueError(f"Frame {i+1} vertex count mismatch: {mesh.vertices.shape} vs {base_vertices.shape}")
    #         if not np.array_equal(mesh.faces, base_mesh.faces):
    #             raise ValueError(f"Topology mismatch in frame {i+1}")
    #         delta = mesh.vertices - base_vertices
    #         morph_deltas.append(delta.astype(np.float32))

    #     morph_deltas = np.array(morph_deltas)  # shape: (num_frames-1, num_vertices, 3)
    #     frame_count = morph_deltas.shape[0] + 1

    # # === Set up GLTF2 ===
    # gltf = GLTF2()
    # gltf.asset = Asset(version="2.0")

    # gltf.buffers = [Buffer(byteLength=0)]
    # gltf.bufferViews = []
    # gltf.accessors = []
    # gltf.meshes = []
    # gltf.nodes = []
    # gltf.scenes = [Scene(nodes=[0])]
    # gltf.scene = 0
    # if not has_animation:
    #     gltf.animations = []

    # buffer_data = bytearray()
    # offset = 0

    # # === Add base position accessor ===
    # base_pos_bytes = base_vertices.astype(np.float32).tobytes()
    # gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_pos_bytes), target=ARRAY_BUFFER))
    # a_pos = len(gltf.accessors)
    # gltf.accessors.append(Accessor(
    #     bufferView=len(gltf.bufferViews)-1,
    #     byteOffset=0,
    #     componentType=FLOAT,
    #     count=len(base_vertices),
    #     type="VEC3"
    # ))
    # # Align to 4 bytes
    # if offset % 4 != 0:
    #     padding = 4 - (offset % 4)
    #     buffer_data += b'\x00' * padding
    #     offset += padding
    # offset += len(base_pos_bytes)
    # buffer_data += base_pos_bytes

    # # === Add base normals accessor ===
    # base_norm_bytes = base_normals.astype(np.float32).tobytes()
    # gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_norm_bytes), target=ARRAY_BUFFER))
    # a_norm = len(gltf.accessors)
    # gltf.accessors.append(Accessor(
    #     bufferView=len(gltf.bufferViews)-1,
    #     byteOffset=0,
    #     componentType=FLOAT,
    #     count=len(base_normals),
    #     type="VEC3"
    # ))
    # # Align to 4 bytes
    # if offset % 4 != 0:
    #     padding = 4 - (offset % 4)
    #     buffer_data += b'\x00' * padding
    #     offset += padding
    # offset += len(base_norm_bytes)
    # buffer_data += base_norm_bytes

    # # === Add base UVs accessor ===
    # base_uv_bytes = base_uvs.astype(np.float32).tobytes()  # UVs as float32
    # gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_uv_bytes), target=ARRAY_BUFFER))
    # a_uv = len(gltf.accessors)
    # gltf.accessors.append(Accessor(
    #     bufferView=len(gltf.bufferViews)-1,
    #     byteOffset=0,
    #     componentType=FLOAT,
    #     count=len(base_uvs),
    #     type="VEC2"
    # ))
    # # Align to 4 bytes
    # if offset % 4 != 0:
    #     padding = 4 - (offset % 4)
    #     buffer_data += b'\x00' * padding
    #     offset += padding
    # offset += len(base_uv_bytes)
    # buffer_data += base_uv_bytes
    
    # # === Add face indices accessor ===
    # face_indices = faces.flatten().astype(np.uint16).tobytes()
    # gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(face_indices), target=ELEMENT_ARRAY_BUFFER))
    # a_indices = len(gltf.accessors)
    # gltf.accessors.append(Accessor(
    #     bufferView=len(gltf.bufferViews)-1,
    #     byteOffset=0,
    #     componentType=UNSIGNED_SHORT,
    #     count=len(faces) * 3,
    #     type="SCALAR"
    # ))
    # # Align to 4 bytes
    # if offset % 4 != 0:
    #     padding = 4 - (offset % 4)
    #     buffer_data += b'\x00' * padding
    #     offset += padding
    # offset += len(face_indices)
    # buffer_data += face_indices

    # # === Add morph target accessors ===
    # morph_targets = []
    # if has_animation :
    #     for delta in morph_deltas:
    #         delta_bytes = delta.tobytes()
    #         gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes), target=ARRAY_BUFFER))
    #         #gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes)))
    #         a_delta = len(gltf.accessors)
    #         min_vals = delta.min(axis=0).tolist()
    #         max_vals = delta.max(axis=0).tolist()
    #         gltf.accessors.append(Accessor(
    #             bufferView=len(gltf.bufferViews)-1,
    #             byteOffset=0,
    #             componentType=FLOAT,
    #             count=len(delta),
    #             type="VEC3",
    #             min=min_vals,
    #             max=max_vals
    #         ))
    #         # Align to 4 bytes
    #         if offset % 4 != 0:
    #             padding = 4 - (offset % 4)
    #             buffer_data += b'\x00' * padding
    #             offset += padding
    #         offset += len(delta_bytes)
    #         buffer_data += delta_bytes
    #         morph_targets.append(a_delta)

    # # === Load texture and normal maps ===
    # image_paths = {
    # "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
    # "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
    # }

    # image_indices = {}

    # for key, path in image_paths.items():
    #     if os.path.isfile(path):
    #         img_data = load_image_bytes(path)
    #         gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(img_data)))
    #         bv_index = len(gltf.bufferViews) - 1

    #         image = Image(uri=None, bufferView=bv_index, mimeType="image/png")
    #         gltf.images.append(image)

    #         texture = Texture(source=len(gltf.images) - 1)
    #         gltf.textures.append(texture)

    #         image_indices[key] = len(gltf.textures) - 1
    #         buffer_data += img_data
    #         # Align to 4 bytes
    #         if offset % 4 != 0:
    #             padding = 4 - (offset % 4)
    #             buffer_data += b'\x00' * padding
    #             offset += padding
    #         offset += len(img_data)
    #     else:
    #         print(f"⚠️ Image missing: {path}")

    #     # === Create material and apply to primitive ===
    # material = Material(
    #     pbrMetallicRoughness=PbrMetallicRoughness(
    #         baseColorTexture=TextureInfo(index=image_indices.get("baseColor"))
    #     ),
    #     normalTexture=TextureInfo(index=image_indices.get("normalMap")) if "normalMap" in image_indices else None
    # )
    # gltf.materials.append(material)
    
    # # === Mesh and Node ===
    # primitive = Primitive(
    #     attributes={
    #     "POSITION": a_pos,
    #     "NORMAL": a_norm,
    #     "TEXCOORD_0": a_uv
    #     },
    #     indices=a_indices,
    #     targets=[{POSITION: a} for a in morph_targets] if has_animation else None,
    #     material = 0
    # )

    # mesh = Mesh(primitives=[primitive], weights=[0.0] * len(morph_targets))
    # gltf.meshes.append(mesh)

    # node = Node(mesh=0)
    # gltf.nodes.append(node)

    # gltf.meshes[0].name = "HeadMesh"
    # gltf.nodes[0].name = "HeadNode"


    # if has_animation :
    #     # === Animation with one channel for all weights ===
    #     # Keyframe times
    #     #time_step = 60.0/ (frame_count-1)
    #     time_data = np.linspace(1 / framerate, frame_count / framerate, num=frame_count - 1, dtype=np.float32)
    #     #time_data = np.arange(frame_count - 1, dtype=np.float32)*time_step
    #     time_bytes = time_data.tobytes()

    #     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(time_bytes)))
    #     a_time = len(gltf.accessors)
    #     gltf.accessors.append(Accessor(
    #         bufferView=len(gltf.bufferViews)-1,
    #         byteOffset=0,
    #         componentType=FLOAT,
    #         count=len(time_data),
    #         type="SCALAR",
    #         min=[float(time_data[0])],
    #         max=[float(time_data[-1])]
    #     ))
    #     # Align to 4 bytes
    #     if offset % 4 != 0:
    #         padding = 4 - (offset % 4)
    #         buffer_data += b'\x00' * padding
    #         offset += padding
    #     offset += len(time_bytes)
    #     buffer_data += time_bytes

    #     # Keyframe weights: each row is a full morph weights vector
    #     weight_data = np.zeros((len(time_data), len(morph_targets)), dtype=np.float32)
    #     for i in range(len(time_data)):
    #         weight_data[i, i] = 1.0  # Only one active morph per frame

    #     weight_bytes = weight_data.astype(np.float32).tobytes()
    #     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(weight_bytes)))
    #     a_weight = len(gltf.accessors)
    #     gltf.accessors.append(Accessor(
    #         bufferView=len(gltf.bufferViews)-1,
    #         byteOffset=0,
    #         componentType=FLOAT,
    #         count=weight_data.size,  # total number of scalars
    #         type="SCALAR"
    #     ))
    #         # Align to 4 bytes
    #     if offset % 4 != 0:
    #         padding = 4 - (offset % 4)
    #         buffer_data += b'\x00' * padding
    #         offset += padding
    #     offset += len(weight_bytes)
    #     buffer_data += weight_bytes

    #     # Single animation sampler + channel
    #     sampler = AnimationSampler(input=a_time, output=a_weight, interpolation="STEP")
    #     channel = AnimationChannel(sampler=0, target={"node": 0, "path": "weights"})

    #     gltf.animations = [Animation(samplers=[sampler], channels=[channel])]

    # # === Finalize buffer ===
    # gltf.buffers[0].byteLength = len(buffer_data)
    # gltf.set_binary_blob(buffer_data)

    # # Check for unintentional type objects before saving
    # find_type_objects(gltf)

    # # Ensure the output directory exists
    # os.makedirs(os.path.dirname(output_glb), exist_ok=True)

    # # Save .glb
    # gltf.save_binary(output_glb)
    # print("✅ Exported to {output_path}")

def export_glb(frame_dir, output_glb, FPS, start_frame=0, end_frame=0):
    print("📦 export_glb called with:")
    print(f"  🔹 frame_dir    = {frame_dir}")
    print(f"  🔹 output_glb   = {output_glb}")
    print(f"  🔹 FPS          = {FPS}")
    print(f"  🔹 start_frame  = {start_frame}")
    print(f"  🔹 end_frame    = {end_frame}")
    
    framerate = FPS
    
    # Get all OBJ frames
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])
    
    # Debug: Show all available frame files
    print("📂 All Frame Paths:")
    for i, path in enumerate(frame_paths):
        print(f"[{i}] {path}")

    if end_frame == 0:
        end_frame = len(frame_paths) - 1

    selected_frame = frame_paths[start_frame:end_frame + 1]

    # Debug: Show selected frame paths
    print("✅ Selected Frames for Export:")
    for i, path in enumerate(selected_frame):
        print(f"[{start_frame + i}] {path}")

    print(f"🧾 Requested Frame Range: start_frame = {start_frame}, end_frame = {end_frame}")
    print(f"🧾 Total frames available: {len(frame_paths)}")

    has_animation = len(selected_frame) > 1
    if not has_animation:
        print("ℹ️ Only one frame detected - exporting static model")

    # Load base mesh
    base_mesh = trimesh.load(selected_frame[0], process=False, maintain_order=True)
    base_vertices = base_mesh.vertices
    base_normals = base_mesh.vertex_normals
    base_uvs = base_mesh.visual.uv
    base_uvs[:, 1] = 1.0 - base_uvs[:, 1]  # Flip V coordinate
    faces = base_mesh.faces

    morph_deltas = []
    if has_animation:
        for i, frame in enumerate(selected_frame[1:]):
            mesh = trimesh.load(frame, process=False, maintain_order=True)
            if mesh.vertices.shape != base_vertices.shape:
                raise ValueError(
                    f"Frame {i+1} vertex count mismatch: "
                    f"{mesh.vertices.shape[0]} vs {base_vertices.shape[0]}"
                )
            if not np.array_equal(mesh.faces, faces):
                raise ValueError(f"Topology mismatch in frame {i+1}")
            delta = mesh.vertices - base_vertices
            morph_deltas.append(delta.astype(np.float32))

        morph_deltas = np.array(morph_deltas)
        frame_count = morph_deltas.shape[0] + 1

    # === Set up GLTF2 ===
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.animations = []

    # Initialize buffers
    gltf.buffers = [Buffer(byteLength=0)]
    gltf.bufferViews = []
    gltf.accessors = []
    gltf.meshes = []
    gltf.nodes = []
    gltf.images = []
    gltf.textures = []
    gltf.materials = []

    buffer_data = bytearray()
    offset = 0

    # === Helper Function for Adding Buffer Data ===
    def add_buffer(data, target=None):
        nonlocal offset, buffer_data
        # Handle both numpy arrays and bytes
        if hasattr(data, 'tobytes'):  # For numpy arrays
            bytes_data = data.tobytes()
        else:  # For raw bytes
            bytes_data = data
        
        # Align to 4 bytes
        padding = (4 - (offset % 4)) % 4
        if padding > 0:
            buffer_data += b'\x00' * padding
            offset += padding
            
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(bytes_data),
            target=target
        ))
        
        buffer_data += bytes_data
        offset += len(bytes_data)
        
        return len(gltf.bufferViews) - 1

    # === Add Geometry Data ===
    # Positions
    bv_pos = add_buffer(base_vertices.astype(np.float32), ARRAY_BUFFER)
    a_pos = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_pos,
        componentType=FLOAT,
        count=len(base_vertices),
        type="VEC3"
    ))
    
    # Normals
    bv_norm = add_buffer(base_normals.astype(np.float32), ARRAY_BUFFER)
    a_norm = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_norm,
        componentType=FLOAT,
        count=len(base_normals),
        type="VEC3"
    ))
    
    # UVs
    bv_uv = add_buffer(base_uvs.astype(np.float32), ARRAY_BUFFER)
    a_uv = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_uv,
        componentType=FLOAT,
        count=len(base_uvs),
        type="VEC2"
    ))
    
    # Faces
    bv_faces = add_buffer(faces.flatten().astype(np.uint16), ELEMENT_ARRAY_BUFFER)
    a_faces = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_faces,
        componentType=UNSIGNED_SHORT,
        count=len(faces) * 3,
        type="SCALAR"
    ))

    # === Add Morph Targets ===
    morph_target_accessors = []
    for delta in morph_deltas:
        bv_delta = add_buffer(delta, ARRAY_BUFFER)
        a_delta = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_delta,
            componentType=FLOAT,
            count=len(delta),
            type="VEC3",
            min=delta.min(axis=0).tolist(),
            max=delta.max(axis=0).tolist()
        ))
        morph_target_accessors.append(a_delta)

    # === Add Textures ===
    #base_name = os.path.splitext(os.path.basename(selected_frame[0]))[0]
    base_name = os.path.splitext(os.path.basename(frame_paths[0]))[0]

    image_paths = {
        "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
        "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
    }

    image_indices = {}
    for key, path in image_paths.items():
        if os.path.isfile(path):
            img_data = load_image_bytes(path)
            bv_img = add_buffer(img_data)
            
            image = Image(bufferView=bv_img, mimeType="image/png")
            gltf.images.append(image)
            
            texture = Texture(source=len(gltf.images)-1)
            gltf.textures.append(texture)
            
            image_indices[key] = len(gltf.textures)-1
        else:
            print(f"⚠️ Image missing: {path}")

    # === Create Material ===
    pbr = PbrMetallicRoughness()
    if "baseColor" in image_indices:
        pbr.baseColorTexture = TextureInfo(index=image_indices["baseColor"])

    material = Material(
        pbrMetallicRoughness=pbr,
        normalTexture=TextureInfo(index=image_indices["normalMap"]) if "normalMap" in image_indices else None
    )
    gltf.materials.append(material)

    # === Create Mesh ===
    primitive = Primitive(
        attributes={
            "POSITION": a_pos,
            "NORMAL": a_norm,
            "TEXCOORD_0": a_uv
        },
        indices=a_faces,
        targets=[{"POSITION": a} for a in morph_target_accessors] if has_animation else None,
        material=0
    )

    mesh = Mesh(
        primitives=[primitive],
        weights=[0.0] * len(morph_target_accessors) if has_animation else None
    )
    gltf.meshes.append(mesh)
    
    node = Node(mesh=0)
    gltf.nodes.append(node)

    # === Create Animation ===
    if has_animation and morph_target_accessors:
        # Time data
        time_data = np.arange(1, frame_count) * (1.0 / framerate)
        bv_time = add_buffer(time_data.astype(np.float32))
        a_time = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_time,
            componentType=FLOAT,
            count=len(time_data),
            type="SCALAR",
            min=[float(time_data[0])],
            max=[float(time_data[-1])]
        ))

        # Weight data
        weight_data = np.zeros((len(time_data), len(morph_target_accessors)), dtype=np.float32)
        for i in range(len(time_data)):
            weight_data[i, i] = 1.0  # Only one active morph per frame
        
        bv_weights = add_buffer(weight_data)
        a_weights = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_weights,
            componentType=FLOAT,
            count=weight_data.size,
            type="SCALAR"
        ))

        # Animation setup
        sampler = AnimationSampler(
            input=a_time,
            output=a_weights,
            interpolation="STEP"
        )
        channel = AnimationChannel(
            sampler=len(gltf.animations[0].samplers) if gltf.animations else 0,
            target={"node": 0, "path": "weights"}
        )
        animation = Animation(samplers=[sampler], channels=[channel])
        gltf.animations.append(animation)

    # === Finalize buffer ===
    gltf.buffers[0].byteLength = len(buffer_data)
    gltf.set_binary_blob(buffer_data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_glb), exist_ok=True)
    gltf.save_binary(output_glb)
    print(f"✅ Exported to {output_glb}")

# def export_glb(frame_dir, output_glb, FPS, start_frame =0, end_frame=0):
#     print("📦 export_glb called with:")
#     print(f"  🔹 frame_dir    = {frame_dir}")
#     print(f"  🔹 output_glb   = {output_glb}")
#     print(f"  🔹 FPS          = {FPS}")
#     print(f"  🔹 start_frame  = {start_frame}")
#     print(f"  🔹 end_frame    = {end_frame}")
#     # === Input Properties ===
#     framerate = FPS
    
#     # === Continue with your original GLB conversion logic using frame_dir ===
#     frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])

#     # Debug: Show all available frame files
#     print("📂 All Frame Paths:")
#     for i, path in enumerate(frame_paths):
#         print(f"[{i}] {path}")

#     if end_frame == 0:
#         end_frame = len(frame_paths) - 1

#     selected_frame = frame_paths [start_frame: end_frame]

#     # Debug: Show selected frame paths
#     print("✅ Selected Frames for Export:")
#     for i, path in enumerate(selected_frame):
#         print(f"[{start_frame + i}] {path}")

#     print(f"🧾 Requested Frame Range: start_frame = {start_frame}, end_frame = {end_frame}")
#     print(f"🧾 Total frames available: {len(frame_paths)}")

#     has_animation = len(selected_frame) > 1
#     #if len(frame_paths) < 2:
#         #raise RuntimeError("Need at least 2 frames for morph animation")
#     if not has_animation:
#         print("ℹ️ Only one frame detected — exporting static model without morph animation.")

#     # Load base mesh and others
#     base_mesh = trimesh.load(selected_frame[0], process=False, maintain_order=True)
#     print("Has visual:", base_mesh.visual.kind)  # should say 'texture'
#     print("UV shape:", base_mesh.visual.uv.shape)
#     base_vertices = base_mesh.vertices
#     base_normals = base_mesh.vertex_normals  # Vertex normals
#     base_uvs = base_mesh.visual.uv  # UV coordinates
#     base_uvs[:,1] = 1.0 - base_uvs[:,1]
#     faces = base_mesh.faces

#     morph_deltas = []
#     if has_animation :
#         for i, frame in enumerate(selected_frame[1:]):
#             mesh = trimesh.load(frame, process=False, maintain_order=True)
#             if mesh.vertices.shape != base_vertices.shape:
#                 raise ValueError(f"Frame {i+1} vertex count mismatch: {mesh.vertices.shape} vs {base_vertices.shape}")
#             if not np.array_equal(mesh.faces, base_mesh.faces):
#                 raise ValueError(f"Topology mismatch in frame {i+1}")
#             delta = mesh.vertices - base_vertices
#             morph_deltas.append(delta.astype(np.float32))

#         morph_deltas = np.array(morph_deltas)  # shape: (num_frames-1, num_vertices, 3)
#         frame_count = morph_deltas.shape[0] + 1

#     # === Set up GLTF2 ===
#     gltf = GLTF2()
#     gltf.asset = Asset(version="2.0")

#     gltf.buffers = [Buffer(byteLength=0)]
#     gltf.bufferViews = []
#     gltf.accessors = []
#     gltf.meshes = []
#     gltf.nodes = []
#     gltf.scenes = [Scene(nodes=[0])]
#     gltf.scene = 0
#     if not has_animation:
#         gltf.animations = []

#     buffer_data = bytearray()
#     offset = 0

#     # === Add base position accessor ===
#     base_pos_bytes = base_vertices.astype(np.float32).tobytes()
#     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_pos_bytes), target=ARRAY_BUFFER))
#     a_pos = len(gltf.accessors)
#     gltf.accessors.append(Accessor(
#         bufferView=len(gltf.bufferViews)-1,
#         byteOffset=0,
#         componentType=FLOAT,
#         count=len(base_vertices),
#         type="VEC3"
#     ))
#     # Align to 4 bytes
#     if offset % 4 != 0:
#         padding = 4 - (offset % 4)
#         buffer_data += b'\x00' * padding
#         offset += padding
#     offset += len(base_pos_bytes)
#     buffer_data += base_pos_bytes

#     # === Add base normals accessor ===
#     base_norm_bytes = base_normals.astype(np.float32).tobytes()
#     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_norm_bytes), target=ARRAY_BUFFER))
#     a_norm = len(gltf.accessors)
#     gltf.accessors.append(Accessor(
#         bufferView=len(gltf.bufferViews)-1,
#         byteOffset=0,
#         componentType=FLOAT,
#         count=len(base_normals),
#         type="VEC3"
#     ))
#     # Align to 4 bytes
#     if offset % 4 != 0:
#         padding = 4 - (offset % 4)
#         buffer_data += b'\x00' * padding
#         offset += padding
#     offset += len(base_norm_bytes)
#     buffer_data += base_norm_bytes

#     # === Add base UVs accessor ===
#     base_uv_bytes = base_uvs.astype(np.float32).tobytes()  # UVs as float32
#     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_uv_bytes), target=ARRAY_BUFFER))
#     a_uv = len(gltf.accessors)
#     gltf.accessors.append(Accessor(
#         bufferView=len(gltf.bufferViews)-1,
#         byteOffset=0,
#         componentType=FLOAT,
#         count=len(base_uvs),
#         type="VEC2"
#     ))
#     # Align to 4 bytes
#     if offset % 4 != 0:
#         padding = 4 - (offset % 4)
#         buffer_data += b'\x00' * padding
#         offset += padding
#     offset += len(base_uv_bytes)
#     buffer_data += base_uv_bytes
    
#     # === Add face indices accessor ===
#     face_indices = faces.flatten().astype(np.uint16).tobytes()
#     gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(face_indices), target=ELEMENT_ARRAY_BUFFER))
#     a_indices = len(gltf.accessors)
#     gltf.accessors.append(Accessor(
#         bufferView=len(gltf.bufferViews)-1,
#         byteOffset=0,
#         componentType=UNSIGNED_SHORT,
#         count=len(faces) * 3,
#         type="SCALAR"
#     ))
#     # Align to 4 bytes
#     if offset % 4 != 0:
#         padding = 4 - (offset % 4)
#         buffer_data += b'\x00' * padding
#         offset += padding
#     offset += len(face_indices)
#     buffer_data += face_indices

#     # === Add morph target accessors ===
#     morph_targets = []
#     if has_animation :
#         for delta in morph_deltas:
#             delta_bytes = delta.tobytes()
#             gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes), target=ARRAY_BUFFER))
#             #gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes)))
#             a_delta = len(gltf.accessors)
#             min_vals = delta.min(axis=0).tolist()
#             max_vals = delta.max(axis=0).tolist()
#             gltf.accessors.append(Accessor(
#                 bufferView=len(gltf.bufferViews)-1,
#                 byteOffset=0,
#                 componentType=FLOAT,
#                 count=len(delta),
#                 type="VEC3",
#                 min=min_vals,
#                 max=max_vals
#             ))
#             # Align to 4 bytes
#             if offset % 4 != 0:
#                 padding = 4 - (offset % 4)
#                 buffer_data += b'\x00' * padding
#                 offset += padding
#             offset += len(delta_bytes)
#             buffer_data += delta_bytes
#             morph_targets.append(a_delta)

#     # === Load texture and normal maps ===
#     base_name = os.path.splitext(os.path.basename(frame_paths[0]))[0]
#     image_paths = {
#     "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
#     "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
#     }

#     image_indices = {}

#     for key, path in image_paths.items():
#         if os.path.isfile(path):
#             img_data = load_image_bytes(path)
#             gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(img_data)))
#             bv_index = len(gltf.bufferViews) - 1

#             image = Image(uri=None, bufferView=bv_index, mimeType="image/png")
#             gltf.images.append(image)

#             texture = Texture(source=len(gltf.images) - 1)
#             gltf.textures.append(texture)

#             image_indices[key] = len(gltf.textures) - 1
#             buffer_data += img_data
#             # Align to 4 bytes
#             if offset % 4 != 0:
#                 padding = 4 - (offset % 4)
#                 buffer_data += b'\x00' * padding
#                 offset += padding
#             offset += len(img_data)
#         else:
#             print(f"⚠️ Image missing: {path}")

#         # === Create material and apply to primitive ===
#     material = Material(
#         pbrMetallicRoughness=PbrMetallicRoughness(
#             baseColorTexture=TextureInfo(index=image_indices.get("baseColor"))
#         ),
#         normalTexture=TextureInfo(index=image_indices.get("normalMap")) if "normalMap" in image_indices else None
#     )
#     gltf.materials.append(material)
    
#     # === Mesh and Node ===
#     primitive = Primitive(
#         attributes={
#         "POSITION": a_pos,
#         "NORMAL": a_norm,
#         "TEXCOORD_0": a_uv
#         },
#         indices=a_indices,
#         targets=[{POSITION: a} for a in morph_targets] if has_animation else None,
#         material = 0
#     )

#     mesh = Mesh(primitives=[primitive], weights=[0.0] * len(morph_targets))
#     gltf.meshes.append(mesh)

#     node = Node(mesh=0)
#     gltf.nodes.append(node)

#     gltf.meshes[0].name = "HeadMesh"
#     gltf.nodes[0].name = "HeadNode"


#     if has_animation :
#         # === Animation with one channel for all weights ===
#         # Keyframe times
#         #time_step = 60.0/ (frame_count-1)
#         time_data = np.linspace(1 / framerate, frame_count / framerate, num=frame_count - 1, dtype=np.float32)
#         #time_data = np.arange(frame_count - 1, dtype=np.float32)*time_step
#         time_bytes = time_data.tobytes()

#         gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(time_bytes)))
#         a_time = len(gltf.accessors)
#         gltf.accessors.append(Accessor(
#             bufferView=len(gltf.bufferViews)-1,
#             byteOffset=0,
#             componentType=FLOAT,
#             count=len(time_data),
#             type="SCALAR",
#             min=[float(time_data[0])],
#             max=[float(time_data[-1])]
#         ))
#         # Align to 4 bytes
#         if offset % 4 != 0:
#             padding = 4 - (offset % 4)
#             buffer_data += b'\x00' * padding
#             offset += padding
#         offset += len(time_bytes)
#         buffer_data += time_bytes

#         # Keyframe weights: each row is a full morph weights vector
#         weight_data = np.zeros((len(time_data), len(morph_targets)), dtype=np.float32)
#         for i in range(len(time_data)):
#             weight_data[i, i] = 1.0  # Only one active morph per frame

#         weight_bytes = weight_data.astype(np.float32).tobytes()
#         gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(weight_bytes)))
#         a_weight = len(gltf.accessors)
#         gltf.accessors.append(Accessor(
#             bufferView=len(gltf.bufferViews)-1,
#             byteOffset=0,
#             componentType=FLOAT,
#             count=weight_data.size,  # total number of scalars
#             type="SCALAR"
#         ))
#             # Align to 4 bytes
#         if offset % 4 != 0:
#             padding = 4 - (offset % 4)
#             buffer_data += b'\x00' * padding
#             offset += padding
#         offset += len(weight_bytes)
#         buffer_data += weight_bytes

#         # Single animation sampler + channel
#         sampler = AnimationSampler(input=a_time, output=a_weight, interpolation="STEP")
#         channel = AnimationChannel(sampler=0, target={"node": 0, "path": "weights"})

#         gltf.animations = [Animation(samplers=[sampler], channels=[channel])]

#     # === Finalize buffer ===
#     gltf.buffers[0].byteLength = len(buffer_data)
#     gltf.set_binary_blob(buffer_data)

#     # Check for unintentional type objects before saving
#     find_type_objects(gltf)

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(output_glb), exist_ok=True)

#     # Save .glb
#     gltf.save_binary(output_glb)
#     print("✅ Exported to {output_path}")

def export_customized_glb(frame_dir, output_glb, FPS, frames, weights):
    print("📦 export_glb called with:")
    print(f"  🔹 frame_dir    = {frame_dir}")
    print(f"  🔹 output_glb   = {output_glb}")
    print(f"  🔹 FPS          = {FPS}")
    print(f"  🔹 frames  = {frames}")
    print(f"  🔹 weights    = {weights}")
    # === Input Properties ===
    framerate = FPS
    
    # === Continue with your original GLB conversion logic using frame_dir ===
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])

    # Debug: Show all available frame files
    print("📂 All Frame Paths:")
    for i, path in enumerate(frame_paths):
        print(f"[{i}] {path}")

    print(f"🧾 Total target morphs available: {len(frame_paths)}")

    has_animation = frames > 1
    #if len(frame_paths) < 2:
        #raise RuntimeError("Need at least 2 frames for morph animation")
    if not has_animation:
        print("ℹ️ Only one frame detected — exporting static model without morph animation.")

    # Load base mesh and others
    base_mesh = trimesh.load(frame_paths[0], process=False, maintain_order=True)
    print("Has visual:", base_mesh.visual.kind)  # should say 'texture'
    print("UV shape:", base_mesh.visual.uv.shape)
    base_vertices = base_mesh.vertices
    base_normals = base_mesh.vertex_normals  # Vertex normals
    base_uvs = base_mesh.visual.uv  # UV coordinates
    base_uvs[:,1] = 1.0 - base_uvs[:,1]
    faces = base_mesh.faces

    morph_deltas = []
    # if has_animation :
    for i, frame in enumerate(frame_paths[1:]):
        mesh = trimesh.load(frame, process=False, maintain_order=True)
        if mesh.vertices.shape != base_vertices.shape:
            raise ValueError(f"Frame {i+1} vertex count mismatch: {mesh.vertices.shape} vs {base_vertices.shape}")
        if not np.array_equal(mesh.faces, base_mesh.faces):
            raise ValueError(f"Topology mismatch in frame {i+1}")
        delta = mesh.vertices - base_vertices
        morph_deltas.append(delta.astype(np.float32))

    morph_deltas = np.array(morph_deltas)  # shape: (num_frames-1, num_vertices, 3)
    frame_count = frames

    # === Set up GLTF2 ===
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")

    gltf.buffers = [Buffer(byteLength=0)]
    gltf.bufferViews = []
    gltf.accessors = []
    gltf.meshes = []
    gltf.nodes = []
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    if not has_animation:
        gltf.animations = []

    buffer_data = bytearray()
    offset = 0

    # === Add base position accessor ===
    base_pos_bytes = base_vertices.astype(np.float32).tobytes()
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_pos_bytes), target=ARRAY_BUFFER))
    a_pos = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews)-1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(base_vertices),
        type="VEC3"
    ))
    # Align to 4 bytes
    if offset % 4 != 0:
        padding = 4 - (offset % 4)
        buffer_data += b'\x00' * padding
        offset += padding
    offset += len(base_pos_bytes)
    buffer_data += base_pos_bytes

    # === Add base normals accessor ===
    base_norm_bytes = base_normals.astype(np.float32).tobytes()
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_norm_bytes), target=ARRAY_BUFFER))
    a_norm = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews)-1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(base_normals),
        type="VEC3"
    ))
    # Align to 4 bytes
    if offset % 4 != 0:
        padding = 4 - (offset % 4)
        buffer_data += b'\x00' * padding
        offset += padding
    offset += len(base_norm_bytes)
    buffer_data += base_norm_bytes

    # === Add base UVs accessor ===
    base_uv_bytes = base_uvs.astype(np.float32).tobytes()  # UVs as float32
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(base_uv_bytes), target=ARRAY_BUFFER))
    a_uv = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews)-1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(base_uvs),
        type="VEC2"
    ))
    # Align to 4 bytes
    if offset % 4 != 0:
        padding = 4 - (offset % 4)
        buffer_data += b'\x00' * padding
        offset += padding
    offset += len(base_uv_bytes)
    buffer_data += base_uv_bytes
    
    # === Add face indices accessor ===
    face_indices = faces.flatten().astype(np.uint16).tobytes()
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(face_indices), target=ELEMENT_ARRAY_BUFFER))
    a_indices = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews)-1,
        byteOffset=0,
        componentType=UNSIGNED_SHORT,
        count=len(faces) * 3,
        type="SCALAR"
    ))
    # Align to 4 bytes
    if offset % 4 != 0:
        padding = 4 - (offset % 4)
        buffer_data += b'\x00' * padding
        offset += padding
    offset += len(face_indices)
    buffer_data += face_indices

    # === Add morph target accessors ===
    morph_targets = []
    #if has_animation :
    for delta in morph_deltas:
        delta_bytes = delta.tobytes()
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes), target=ARRAY_BUFFER))
        #gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes)))
        a_delta = len(gltf.accessors)
        min_vals = delta.min(axis=0).tolist()
        max_vals = delta.max(axis=0).tolist()
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews)-1,
            byteOffset=0,
            componentType=FLOAT,
            count=len(delta),
            type="VEC3",
            min=min_vals,
            max=max_vals
        ))
        # Align to 4 bytes
        if offset % 4 != 0:
            padding = 4 - (offset % 4)
            buffer_data += b'\x00' * padding
            offset += padding
        offset += len(delta_bytes)
        buffer_data += delta_bytes
        morph_targets.append(a_delta)

    # === Load texture and normal maps ===
    base_name = os.path.splitext(os.path.basename(frame_paths[0]))[0]
    image_paths = {
    "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
    "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
    }

    image_indices = {}

    for key, path in image_paths.items():
        if os.path.isfile(path):
            img_data = load_image_bytes(path)
            gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(img_data)))
            bv_index = len(gltf.bufferViews) - 1

            image = Image(uri=None, bufferView=bv_index, mimeType="image/png")
            gltf.images.append(image)

            texture = Texture(source=len(gltf.images) - 1)
            gltf.textures.append(texture)

            image_indices[key] = len(gltf.textures) - 1
            buffer_data += img_data
            # Align to 4 bytes
            if offset % 4 != 0:
                padding = 4 - (offset % 4)
                buffer_data += b'\x00' * padding
                offset += padding
            offset += len(img_data)
        else:
            print(f"⚠️ Image missing: {path}")

        # === Create material and apply to primitive ===
    material = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=image_indices.get("baseColor"))
        ),
        normalTexture=TextureInfo(index=image_indices.get("normalMap")) if "normalMap" in image_indices else None
    )
    gltf.materials.append(material)
    
    # === Mesh and Node ===
    primitive = Primitive(
        attributes={
        "POSITION": a_pos,
        "NORMAL": a_norm,
        "TEXCOORD_0": a_uv
        },
        indices=a_indices,
        targets=[{POSITION: a} for a in morph_targets] if has_animation else None,
        material = 0
    )

        # Validate weights
    if len(weights) != len(morph_targets):
        raise ValueError("Length of weights must match number of morph targets.")

    mesh = Mesh(primitives=[primitive], weights=[0.0] * len(morph_targets)if has_animation else weights)
    gltf.meshes.append(mesh)

    node = Node(mesh=0)
    gltf.nodes.append(node)

    gltf.meshes[0].name = "HeadMesh"
    gltf.nodes[0].name = "HeadNode"

    if has_animation :
        # === Animation with one channel for all weights ===
        # Keyframe times
        #time_step = 60.0/ (frame_count-1)
        # Generate time values as integer multiples of the frame duration
        frame_duration = 1.0 / framerate  # Time per frame in seconds
        time_data = np.arange(1, frame_count) * frame_duration  # [0.0, 0.033, 0.066, ...]
        time_data = time_data.astype(np.float32)  # Ensure float32 precision

        # Validate no negative or corrupted values
        assert np.all(time_data >= 0), "Negative time values detected!"
        # time_data = np.linspace(1 / framerate, frame_count / framerate, num=frame_count - 1, dtype=np.float32)
        print("Generated time_data:", time_data)
        #time_data = np.arange(frame_count - 1, dtype=np.float32)*time_step
        time_bytes = time_data.tobytes()

        # Convert the entire byte array back into float32 values
        all_time_values = np.frombuffer(time_bytes, dtype=np.float32)

        # Print all the values
        print("All time values from bytes:", all_time_values)

        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(time_bytes)))
        a_time = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews)-1,
            byteOffset=0,
            componentType=FLOAT,
            count=len(time_data),
            type="SCALAR",
            min=[float(time_data[0])],
            max=[float(time_data[-1])]
        ))
        # Align to 4 bytes
        if offset % 4 != 0:
            padding = 4 - (offset % 4)
            buffer_data += b'\x00' * padding
            offset += padding
        offset += len(time_bytes)
        buffer_data += time_bytes

        # Keyframe weights: each row is a full morph weights vector
        # weight_data = np.zeros((len(time_data), len(morph_targets)), dtype=np.float32)
        # for i in range(len(time_data)):
        #     weight_data[i, i] = 1.0  # Only one active morph per frame

        # Interpolated weight keyframes
        # keyframes = []
        # for t in range(frames):
        #     alpha = t / (frames - 1)
        #     keyframes.append([w * alpha for w in weights])
        # weight_data = np.array(keyframes, dtype=np.float32)
        
        # Keyframe weights: each row is a full morph weights vector
        weight_data = np.zeros((len(time_data), len(morph_targets)), dtype=np.float32)

        for i, w in enumerate(weights):
            for t in range(len(time_data)):
                interp = (t + 1) / len(time_data)  # t from 0 to len-1
                weight_data[t, i] = w * interp  # linearly interpolate from 0 to weight

        print("time_data:", time_data)
        print(f"weight_data shape: {weight_data.shape}")
        print("weight_data:\n", weight_data)
        weight_bytes = weight_data.astype(np.float32).tobytes()
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(weight_bytes)))
        a_weight = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews)-1,
            byteOffset=0,
            componentType=FLOAT,
            count=weight_data.size,  # total number of scalars
            type="SCALAR"
        ))
            # Align to 4 bytes
        if offset % 4 != 0:
            padding = 4 - (offset % 4)
            buffer_data += b'\x00' * padding
            offset += padding
        offset += len(weight_bytes)
        buffer_data += weight_bytes

        # Single animation sampler + channel
        sampler = AnimationSampler(input=a_time, output=a_weight, interpolation="STEP")
        channel = AnimationChannel(sampler=0, target={"node": 0, "path": "weights"})

        gltf.animations = [Animation(samplers=[sampler], channels=[channel])]

    # === Finalize buffer ===
    gltf.buffers[0].byteLength = len(buffer_data)
    gltf.set_binary_blob(buffer_data)

    # Check for unintentional type objects before saving
    find_type_objects(gltf)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_glb), exist_ok=True)

    # Save .glb
    gltf.save_binary(output_glb)
    print("✅ Exported to {output_path}")

    print("Time accessor data:", gltf.accessors[a_time].__dict__)
    print("Weight accessor data:", gltf.accessors[a_weight].__dict__)

    for i, acc in enumerate(gltf.accessors):
        print(f"[{i}] Accessor type={acc.type}, count={acc.count}, min={acc.min}, max={acc.max}")

def export_animated_glb(frame_dir, output_glb, FPS, frames, target_weights):
    """
    Export a GLB with animation from neutral to target expression
    
    Args:
        frame_dir: Directory containing OBJ frames and textures
        output_glb: Output GLB file path
        FPS: Frames per second for animation
        frames: Number of frames in animation
        target_weights: List of target weights for each morph target (e.g., [0.5, 0.8, 0.3])
    """
    print("🎬 export_animated_glb called with:")
    print(f"  🔹 frame_dir      = {frame_dir}")
    print(f"  🔹 output_glb     = {output_glb}")
    print(f"  🔹 FPS            = {FPS}")
    print(f"  🔹 frames         = {frames}")
    print(f"  🔹 target_weights = {target_weights}")
    
    # === Input Validation ===
    if frames < 1:
        raise ValueError("Need at least 1 frame")
    
    # === Load Frames ===
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])
    print(f"🧾 Total morph targets available: {len(frame_paths)}")
    
    if len(target_weights) != len(frame_paths) - 1:
        raise ValueError(f"Length of target_weights ({len(target_weights)}) must match number of morph targets ({len(frame_paths) - 1})")

    # === Load Base Mesh ===
    base_mesh = trimesh.load(frame_paths[0], process=False, maintain_order=True)
    base_vertices = base_mesh.vertices
    base_normals = base_mesh.vertex_normals
    base_uvs = base_mesh.visual.uv
    base_uvs[:,1] = 1.0 - base_uvs[:,1]  # Flip UV Y-coordinate
    faces = base_mesh.faces

    # === Calculate Morph Deltas ===
    morph_deltas = []
    for frame in frame_paths[1:]:
        mesh = trimesh.load(frame, process=False, maintain_order=True)
        if mesh.vertices.shape != base_vertices.shape:
            raise ValueError(f"Vertex count mismatch in frame {frame}")
        if not np.array_equal(mesh.faces, base_mesh.faces):
            raise ValueError(f"Topology mismatch in frame {frame}")
        delta = mesh.vertices - base_vertices
        morph_deltas.append(delta.astype(np.float32))
    
    morph_deltas = np.array(morph_deltas)  # shape: (num_morph_targets, num_vertices, 3)

    # === Set up GLTF ===
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")
    gltf.buffers = [Buffer(byteLength=0)]
    gltf.bufferViews = []
    gltf.accessors = []
    gltf.meshes = []
    gltf.nodes = []
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.animations = []

    buffer_data = bytearray()
    offset = 0

    # === Helper Function for Adding Buffer Data ===
    def add_buffer(data, target=None):
        nonlocal offset, buffer_data
        # Handle both numpy arrays and bytes
        if hasattr(data, 'tobytes'):  # For numpy arrays
            bytes_data = data.tobytes()
        else:  # For raw bytes
            bytes_data = data
        
        # Align to 4 bytes
        padding = (4 - (offset % 4)) % 4
        if padding > 0:
            buffer_data += b'\x00' * padding
            offset += padding
            
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(bytes_data),
            target=target
        ))
        
        buffer_data += bytes_data
        offset += len(bytes_data)
        
        return len(gltf.bufferViews) - 1

    # === Add Geometry Data ===
    # Positions
    bv_pos = add_buffer(base_vertices.astype(np.float32), ARRAY_BUFFER)
    a_pos = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_pos,
        componentType=FLOAT,
        count=len(base_vertices),
        type="VEC3"
    ))
    
    # Normals
    bv_norm = add_buffer(base_normals.astype(np.float32), ARRAY_BUFFER)
    a_norm = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_norm,
        componentType=FLOAT,
        count=len(base_normals),
        type="VEC3"
    ))
    
    # UVs
    bv_uv = add_buffer(base_uvs.astype(np.float32), ARRAY_BUFFER)
    a_uv = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_uv,
        componentType=FLOAT,
        count=len(base_uvs),
        type="VEC2"
    ))
    
    # Faces
    bv_faces = add_buffer(faces.flatten().astype(np.uint16), ELEMENT_ARRAY_BUFFER)
    a_faces = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=bv_faces,
        componentType=UNSIGNED_SHORT,
        count=len(faces) * 3,
        type="SCALAR"
    ))

    # === Add Morph Targets ===
    morph_target_accessors = []
    for delta in morph_deltas:
        bv_delta = add_buffer(delta, ARRAY_BUFFER)
        a_delta = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_delta,
            componentType=FLOAT,
            count=len(delta),
            type="VEC3",
            min=delta.min(axis=0).tolist(),
            max=delta.max(axis=0).tolist()
        ))
        morph_target_accessors.append(a_delta)

    # === Add Textures ===
    base_name = os.path.splitext(os.path.basename(frame_paths[0]))[0]
    image_paths = {
        "baseColor": os.path.join(frame_dir, f"{base_name}.png"),
        "normalMap": os.path.join(frame_dir, f"{base_name}_normals.png")
    }

    image_indices = {}
    for key, path in image_paths.items():
        if os.path.isfile(path):
            img_data = load_image_bytes(path)
            bv_img = add_buffer(img_data)
            
            image = Image(bufferView=bv_img, mimeType="image/png")
            gltf.images.append(image)
            
            texture = Texture(source=len(gltf.images)-1)
            gltf.textures.append(texture)
            
            image_indices[key] = len(gltf.textures)-1
        else:
            print(f"⚠️ Image missing: {path}")

    # === Create Material ===
    material = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=image_indices.get("baseColor"))
        ),
        normalTexture=TextureInfo(index=image_indices.get("normalMap")) if "normalMap" in image_indices else None
    )
    gltf.materials.append(material)

    # === Create Mesh ===
    primitive = Primitive(
        attributes={
            "POSITION": a_pos,
            "NORMAL": a_norm,
            "TEXCOORD_0": a_uv
        },
        indices=a_faces,
        targets=[{"POSITION": a} for a in morph_target_accessors],
        material=0
    )
    
    # Set initial weights based on frames
    if frames == 1:
        # Static model with target weights applied directly
        initial_weights = target_weights
        print("ℹ️ Creating static model with target weights applied")
    else:
        # Animated model starting from neutral
        initial_weights = [0.0] * len(morph_target_accessors)
        print("ℹ️ Creating animated model transitioning to target weights")

    mesh = Mesh(
        primitives=[primitive],
        weights=initial_weights
    )
    gltf.meshes.append(mesh)
    
    node = Node(mesh=0)
    gltf.nodes.append(node)

    # === Create Animation (only if frames > 1) ===
    if frames > 1:
        # Time data (0 to duration in seconds)
        duration = (frames - 1) / FPS
        time_data = np.linspace(1/FPS, duration, num=frames, dtype=np.float32)
        bv_time = add_buffer(time_data)
        a_time = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_time,
            componentType=FLOAT,
            count=len(time_data),
            type="SCALAR",
            min=[1/FPS],
            max=[float(duration)]
        ))
        
        # Weight data - interpolate from 0 to target weights
        weight_data = np.zeros((frames, len(morph_target_accessors)), dtype=np.float32)
        for i in range(frames):
            t = i / (frames - 1)  # Normalized time [0, 1]
            weight_data[i] = [t * w for w in target_weights]
        
        bv_weights = add_buffer(weight_data)
        a_weights = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_weights,
            componentType=FLOAT,
            count=weight_data.size,
            type="SCALAR"
        ))
        
        # Create animation sampler and channel
        sampler = AnimationSampler(
            input=a_time,
            output=a_weights,
            interpolation="LINEAR"  # Smooth transition
        )
        
        channel = AnimationChannel(
            sampler=0,
            target={"node": 0, "path": "weights"}
        )
        
        gltf.animations.append(Animation(
            samplers=[sampler],
            channels=[channel]
        ))

    # === Finalize and Export ===
    gltf.buffers[0].byteLength = len(buffer_data)
    gltf.set_binary_blob(buffer_data)
    
    os.makedirs(os.path.dirname(output_glb), exist_ok=True)
    gltf.save_binary(output_glb)
    print(f"✅ Exported animated GLB to {output_glb}")

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .obj sequence to morph-target .glb animation")
    parser.add_argument('--input_root', type=str, required=True, help='Base directory containing folders with obj frames')
    parser.add_argument('--frame_dir', type=str, required=True, help='Temporary directory to gather .obj frames for GLB conversion')
    parser.add_argument('--output_glb', type=str, required=True, help='Path to save the output .glb')
    parser.add_argument('--FPS', type=float, default=30.0, help='Animation framerate (default: 30.0)')
    args = parser.parse_args()
    
    main(args.input_root, args.frame_dir, args.output_glb, args.FPS)