import os
import trimesh
import numpy as np
#from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Animation, AnimationChannel, AnimationSampler, BufferTarget, FLOAT, UNSIGNED_SHORT, Position, Weights, MorphTarget, MeshPrimitive, PERSPECTIVE
from pygltflib import *

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

# === Load .obj sequence ===
frame_dir = "TestSamples/animation_results/video6323231714244041546"
frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".obj")])

if len(frame_paths) < 2:
    raise RuntimeError("Need at least 2 frames for morph animation")

# Load base mesh and others
base_mesh = trimesh.load(frame_paths[0], process=False)
base_vertices = base_mesh.vertices
faces = base_mesh.faces

morph_deltas = []
for i, frame in enumerate(frame_paths[1:]):
    mesh = trimesh.load(frame, process=False)
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

# === Mesh and Node ===
primitive = Primitive(
    attributes={"POSITION": a_pos},
    indices=a_indices,
    targets=[{POSITION: a} for a in morph_targets]
)

mesh = Mesh(primitives=[primitive], weights=[0.0] * len(morph_targets))
gltf.meshes.append(mesh)

node = Node(mesh=0)
gltf.nodes.append(node)

# === Animation with one channel for all weights ===
# Keyframe times
time_data = np.arange(frame_count - 1, dtype=np.float32)
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

# Save .glb
gltf.save_binary("TestSamples/animation_results/video6323231714244041546/output_animation.glb")
print("✅ Exported to output_animation.glb")
