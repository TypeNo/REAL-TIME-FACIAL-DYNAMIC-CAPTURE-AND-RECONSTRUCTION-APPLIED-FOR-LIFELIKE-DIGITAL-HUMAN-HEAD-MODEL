
import trimesh
import numpy as np
from pygltflib import *
import os
from pathlib import Path
import json

FLOAT = 5126
UNSIGNED_SHORT = 5123
ARRAY_BUFFER = 34962  # eg vertex data
ELEMENT_ARRAY_BUFFER = 34963  # eg index data

POSITION = "POSITION"
NORMAL = "NORMAL"
TANGENT = "TANGENT"
TEXCOORD_0 = "TEXCOORD_0"
TEXCOORD_1 = "TEXCOORD_1"
COLOR_0 = "COLOR_0"
JOINTS_0 = "JOINTS_0"
WEIGHTS_0 = "WEIGHTS_0"

SCALAR = "SCALAR"
VEC2 = "VEC2"
VEC3 = "VEC3"
VEC4 = "VEC4"
MAT2 = "MAT2"
MAT3 = "MAT3"
MAT4 = "MAT4"


# CONFIGURATION
#obj_dir = Path("obj_frames")  # Directory with obj files
obj_dir = Path("TestSamples/animation_results/video6323231714244041546_frame0000")  # Directory with obj files
texture_path = "video6323231714244041546_frame0000.png"
normal_map_path = "video6323231714244041546_frame0000_normals.png"
frame_count = 31
output_path = "TestSamples/animation_results/video6323231714244041546_frame0000/animated_model.glb"
name_prefix = "video6323231714244041546_frame"

# Load base mesh (first frame)
base_mesh = trimesh.load(obj_dir / f"{name_prefix}0000.obj", process=False)
base_vertices = base_mesh.vertices
indices = base_mesh.faces.flatten()
texcoords = base_mesh.visual.uv

# Compute morph targets (vertex deltas)
morph_deltas = []
for i in range(1, frame_count):
    mesh = trimesh.load(obj_dir / f"{name_prefix}{i:04d}.obj", process=False)
    delta = mesh.vertices - base_vertices
    morph_deltas.append(delta.astype(np.float32))

# Flatten data
flat_vertices = base_vertices.flatten().astype(np.float32).tobytes()
flat_normals = base_mesh.vertex_normals.flatten().astype(np.float32).tobytes()
flat_texcoords = texcoords.flatten().astype(np.float32).tobytes()
flat_indices = indices.astype(np.uint16).tobytes()
morph_targets_bytes = [delta.flatten().tobytes() for delta in morph_deltas]

# GLTF Setup
gltf = GLTF2(asset=Asset(version="2.0"))

# Create binary blob for buffers
buffer_data = flat_vertices + flat_normals + flat_texcoords + flat_indices + b"".join(morph_targets_bytes)
buffer = Buffer(byteLength=len(buffer_data))
gltf.buffers.append(buffer)

# Create bufferView for each chunk
offset = 0

def add_buffer_view(data, target):
    global offset
    view = BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target)
    gltf.bufferViews.append(view)
    old_offset = offset
    offset += len(data)
    return len(gltf.bufferViews) - 1, old_offset

# Add all bufferViews
bv_pos, off_pos = add_buffer_view(flat_vertices, ARRAY_BUFFER)
bv_nrm, off_nrm = add_buffer_view(flat_normals, ARRAY_BUFFER)
bv_tex, off_tex = add_buffer_view(flat_texcoords, ARRAY_BUFFER)
bv_idx, off_idx = add_buffer_view(flat_indices, ELEMENT_ARRAY_BUFFER)
bv_morphs = []
for data in morph_targets_bytes:
    idx, _ = add_buffer_view(data, ARRAY_BUFFER)
    bv_morphs.append(idx)

# Accessors
def add_accessor(bv_idx, offset, count, type_, componentType, max_, min_):
    acc = Accessor(bufferView=bv_idx, byteOffset=offset, componentType=componentType,
                   count=count, type=type_, max=max_, min=min_)
    gltf.accessors.append(acc)
    return len(gltf.accessors) - 1

v_count = len(base_vertices)
idx_count = len(indices)
a_pos = add_accessor(bv_pos, 0, v_count, VEC3, FLOAT, base_vertices.max(0).tolist(), base_vertices.min(0).tolist())
a_nrm = add_accessor(bv_nrm, 0, v_count, VEC3, FLOAT, [1,1,1], [-1,-1,-1])
a_tex = add_accessor(bv_tex, 0, v_count, VEC2, FLOAT, [1,1], [0,0])
a_idx = add_accessor(bv_idx, 0, idx_count, SCALAR, UNSIGNED_SHORT, [int(indices.max())], [int(indices.min())])
a_morphs = [add_accessor(i, 0, v_count, VEC3, FLOAT, [1,1,1], [-1,-1,-1]) for i in bv_morphs]

# Create mesh
mesh = Mesh(primitives=[Primitive(attributes={POSITION: a_pos, NORMAL: a_nrm, TEXCOORD_0: a_tex},
                                      indices=a_idx,
                                      targets=[{POSITION: a} for a in a_morphs])])
gltf.meshes.append(mesh)

# Create material
material = Material(name="TexturedMaterial", pbrMetallicRoughness=PbrMetallicRoughness())
gltf.materials.append(material)

# Create node and scene
gltf.nodes.append(Node(mesh=0))
gltf.scenes.append(Scene(nodes=[0]))
gltf.scene = 0

# Create animation using morph weights
sampler = AnimationSampler(input=0, output=1)
channel = AnimationChannel(sampler=0, target=AnimationChannelTarget(node=0, path="weights"))
gltf.animations.append(Animation(samplers=[sampler], channels=[channel]))

# Animation accessors (times + weights)
times = np.arange(0, frame_count-1, dtype=np.float32)
#weights = np.eye(frame_count, dtype=np.float32)
weights = np.eye(frame_count-1, dtype=np.float32)
print(f"weights shape: {weights.shape}, morph_targets_bytes length: {len(morph_targets_bytes)}")

gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=times.nbytes))
a_time = len(gltf.accessors)
gltf.accessors.append(Accessor(bufferView=len(gltf.bufferViews)-1, byteOffset=0, componentType=FLOAT,
                               count=len(times), type=SCALAR, max=[float(times[-1])], min=[0]))
offset += times.nbytes
buffer_data += times.tobytes()

gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=weights.nbytes))
a_weight = len(gltf.accessors)
gltf.accessors.append(Accessor(bufferView=len(gltf.bufferViews)-1, byteOffset=0, componentType=FLOAT,
                               count=len(weights), type=f"VEC{weights.shape[1]}"))
offset += weights.nbytes
buffer_data += weights.tobytes()

gltf.animations[0].samplers[0].input = a_time
gltf.animations[0].samplers[0].output = a_weight

# Final buffer binary
gltf.set_binary_blob(buffer_data)

def find_type_objects(obj, path="root"):
    if isinstance(obj, type):
        print(f"⚠️ Type object found at: {path} -> {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_type_objects(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_type_objects(item, f"{path}[{i}]")
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            find_type_objects(v, f"{path}.{k}")


find_type_objects(gltf)


try:
    json_str = gltf.to_json(default=lambda x: f"<<non-serializable: {type(x)}>>")
    print("GLTF is serializable.")
except TypeError as e:
    print(f"Serialization error: {e}")


gltf.save_binary(output_path)
