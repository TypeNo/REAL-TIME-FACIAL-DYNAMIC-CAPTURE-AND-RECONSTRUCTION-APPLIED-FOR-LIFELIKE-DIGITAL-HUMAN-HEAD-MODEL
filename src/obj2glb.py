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
    input_root = input_root
    frame_dir = frame_dir
    os.makedirs(frame_dir, exist_ok=True)

    # === Gather all .obj files from subfolders ===
    all_obj_paths = glob.glob(os.path.join(input_root, "**", "*.obj"), recursive=True)
    obj_paths = sorted([p for p in all_obj_paths if not p.endswith("_detail.obj")])

    # Optionally: Copy them into frame_dir for clean naming
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

    # === Copy texture and normal map if they exist ===
    for file_path in [texture_path, normal_map_path]:
        if os.path.isfile(file_path):
            shutil.copy2(file_path, os.path.join(frame_dir, os.path.basename(file_path)))
            print(f"✅ Copied: {file_path}")
        else:
            print(f"⚠️ Missing file: {file_path}")
    
    # === Continue with your original GLB conversion logic using frame_dir ===
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])
    has_animation = len(frame_paths) > 1
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
    if has_animation :
        for i, frame in enumerate(frame_paths[1:]):
            mesh = trimesh.load(frame, process=False, maintain_order=True)
            delta = mesh.vertices - base_vertices
            morph_deltas.append(delta.astype(np.float32))

        morph_deltas = np.array(morph_deltas)  # shape: (num_frames-1, num_vertices, 3)
        frame_count = morph_deltas.shape[0] + 1

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
    offset += len(face_indices)
    buffer_data += face_indices

    # === Add morph target accessors ===
    morph_targets = []
    if has_animation :
        for delta in morph_deltas:
            delta_bytes = delta.tobytes()
            gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(delta_bytes), target=ARRAY_BUFFER))
            a_delta = len(gltf.accessors)
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews)-1,
                byteOffset=0,
                componentType=FLOAT,
                count=len(delta),
                type="VEC3"
            ))
            offset += len(delta_bytes)
            buffer_data += delta_bytes
            morph_targets.append(a_delta)

    # === Load texture and normal maps ===
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

    mesh = Mesh(primitives=[primitive], weights=[0.0] * len(morph_targets))
    gltf.meshes.append(mesh)

    node = Node(mesh=0)
    gltf.nodes.append(node)

    if has_animation :
        # === Animation with one channel for all weights ===
        # Keyframe times
        #time_step = 60.0/ (frame_count-1)
        time_data = np.linspace(1 / framerate, frame_count / framerate, num=frame_count - 1, dtype=np.float32)
        #time_data = np.arange(frame_count - 1, dtype=np.float32)*time_step
        time_bytes = time_data.tobytes()

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
        offset += len(time_bytes)
        buffer_data += time_bytes

        # Keyframe weights: each row is a full morph weights vector
        weight_data = np.zeros((len(time_data), len(morph_targets)), dtype=np.float32)
        for i in range(len(time_data)):
            weight_data[i, i] = 1.0  # Only one active morph per frame

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

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .obj sequence to morph-target .glb animation")
    parser.add_argument('--input_root', type=str, required=True, help='Base directory containing folders with obj frames')
    parser.add_argument('--frame_dir', type=str, required=True, help='Temporary directory to gather .obj frames for GLB conversion')
    parser.add_argument('--output_glb', type=str, required=True, help='Path to save the output .glb')
    parser.add_argument('--FPS', type=float, default=30.0, help='Animation framerate (default: 30.0)')
    args = parser.parse_args()
    
    main(args.input_root, args.frame_dir, args.output_glb, args.FPS)