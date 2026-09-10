#!/usr/bin/env python3
"""Capture same-clip motions and timings for the comparison HTML."""
from __future__ import annotations
import json, math, os, subprocess, time
from pathlib import Path
import numpy as np

OUT = Path("evidence/compare")
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "walk forward\n"
FRAMES = 30
STEPS = 10
SEED = 42
FPS = 30.0

PARENTS = [-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28]
NAMES = ["Hips","Spine1","Spine2","Chest","Neck1","Neck2","Head","Jaw","LeftEye","RightEye","LeftShoulder","LeftArm","LeftForeArm","LeftHand","LeftHandThumbEnd","LeftHandMiddleEnd","RightShoulder","RightArm","RightForeArm","RightHand","RightHandThumbEnd","RightHandMiddleEnd","LeftLeg","LeftShin","LeftFoot","LeftToeBase","RightLeg","RightShin","RightFoot","RightToeBase"]
OFFSETS = np.array([
 [0,0,0],[-0.00013727,0.0500376256,-0.00053726669],[0,0.0712530139,-0.000298248546],
 [0,0.0755006305,-0.00815970992],[-0.00181676517,0.263112953,-0.00553348292],
 [0,0.0770939664,0.0230258546],[0,0.0612891595,0.0195370861],[0,0.0047559225,0.0309494062],
 [0.0320638079,0.0538020513,0.0758688308],[-0.0322244017,0.05361869,0.0755823359],
 [0.0162165175,0.232371641,0.0511341324],[0.149198457,0,-0.0550232576],
 [0.287393078,0,0],[0.270939812,0,0],[0.122686267,-0.0322017573,0.0483306876],
 [0.190119595,-0.00312878387,-0.000339570373],[-0.0138011824,0.231803086,0.0521415786],
 [-0.150371962,0,-0.0554560437],[-0.287366393,0,0],[-0.271336198,0,0],
 [-0.122642483,-0.0321145448,0.0480403904],[-0.190005945,-0.00306615542,-0.0003157343],
 [0.10043214,-0.0843452671,0.0259565473],[0,-0.432217537,-0.00802912805],
 [0,-0.421550959,-0.0348152298],[0,-0.0505947206,0.132315294],
 [-0.10047278,-0.0829525995,0.0262031695],[0,-0.433622059,-0.00805555828],
 [0,-0.421173943,-0.0347839785],[0,-0.0507960932,0.132841956],
], dtype=np.float32)
BONES = [[i,p] for i,p in enumerate(PARENTS) if p>=0]

def quat_to_mat(q):
    x,y,z,w = [float(v) for v in q]
    n = math.sqrt(x*x+y*y+z*z+w*w) or 1.0
    x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

def fk(rot, root):
    frames, joints, _ = rot.shape
    posed = np.zeros((frames, joints, 3), dtype=np.float32)
    glob = np.zeros((frames, joints, 3, 3), dtype=np.float32)
    for t in range(frames):
        for j, parent in enumerate(PARENTS):
            local = quat_to_mat(rot[t,j])
            if parent < 0:
                glob[t,j] = local
                posed[t,j] = root[t]
            else:
                glob[t,j] = glob[t,parent] @ local
                posed[t,j] = posed[t,parent] + glob[t,parent] @ OFFSETS[j]
    return posed

def write_clip(name, rot, root, extra=None):
    posed = fk(rot, root)
    payload = {
        "name": name, "fps": FPS, "frames": int(rot.shape[0]), "joints": int(rot.shape[1]),
        "names": NAMES, "parents": PARENTS, "bones": BONES,
        "root": root.astype(np.float32).tolist(),
        "posed": posed.astype(np.float32).tolist(),
        "extra": extra or {},
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path, posed, root

def capture_mlx(rows):
    from kimodo_mlx.motion import MotionModel
    from kimodo_mlx.text_encoder import TextEncoder
    enc = TextEncoder(Path("models/llm2vec-text-bundle/generated/llm2vec-text-bundle"))
    t0=time.perf_counter(); embed=enc.encode(PROMPT); cold_enc=time.perf_counter()-t0
    t1=time.perf_counter(); embed2=enc.encode(PROMPT); warm_enc=time.perf_counter()-t1
    model=MotionModel.load("models/nvidia-soma-rp-v1.1")
    t2=time.perf_counter(); r=model.sample(embed, frames=FRAMES, steps=STEPS, seed=SEED); cold_s=time.perf_counter()-t2
    t3=time.perf_counter(); r2=model.sample(embed, frames=FRAMES, steps=STEPS, seed=SEED); warm_s=time.perf_counter()-t3
    write_clip("mlx-metal", r.local_rotations_xyzw, r.root_positions, {"seed": SEED, "deterministic": bool(np.allclose(r.local_rotations_xyzw, r2.local_rotations_xyzw))})
    rows.append({"id":"mlx-cold","label":"kimodo-mlx (Metal, cold)","backend":"mlx-metal","weight_resident":False,"encode_s":cold_enc,"sample_s":cold_s,"e2e_s":cold_enc+cold_s,"runnable":True,"notes":"First encode reads ~15GB LLM2Vec GGUF into unified memory."})
    rows.append({"id":"mlx-warm","label":"kimodo-mlx (Metal, weights resident)","backend":"mlx-metal","weight_resident":True,"encode_s":warm_enc,"sample_s":warm_s,"e2e_s":warm_enc+warm_s,"runnable":True,"notes":"Second encode/sample with resident weights. Motion is deterministic for a fixed seed."})

def capture_cpp(rows, backend, name):
    binary=Path(".cache/kimodo-build-metal/kmd-generate")
    prompt=Path("/tmp/kimodo-compare-prompt.txt"); prompt.write_text(PROMPT)
    out=Path(f"/tmp/kimodo-compare-{name}"); out.mkdir(exist_ok=True)
    env=os.environ.copy(); env["KIMODO_BACKEND"]=backend
    cmd=[str(binary),"models/models/kimodo-soma-rp-v1.1-f32.gguf","models/llm2vec-text-bundle/generated/llm2vec-text-bundle",str(prompt),str(FRAMES),str(STEPS),str(SEED),str(out)]
    t0=time.perf_counter(); proc=subprocess.run(cmd, env=env, capture_output=True, text=True); e2e=time.perf_counter()-t0
    if proc.returncode!=0:
        rows.append({"id":name,"label":f"kimodo.cpp {backend}","backend":backend,"weight_resident":False,"e2e_s":e2e,"runnable":False,"notes":(proc.stderr or proc.stdout)[-400:]})
        return
    rot=np.fromfile(out/"local_rotations_xyzw.f32", dtype=np.float32).reshape(FRAMES,30,4)
    root=np.fromfile(out/"root_positions.f32", dtype=np.float32).reshape(FRAMES,3)
    write_clip(name, rot, root)
    rows.append({"id":name,"label":f"kimodo.cpp ({backend})","backend":backend,"weight_resident":False,"encode_s":None,"sample_s":None,"e2e_s":e2e,"runnable":True,"notes":"Native C++ generate includes text encode + motion sample. Reloads the text bundle each process."})

def probe_onnx(rows):
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is None:
        rows.append({"id":"onnx","label":"ONNX Runtime / CoreML","backend":"onnx","runnable":False,"e2e_s":None,"notes":"onnxruntime is not installed and no Kimodo ONNX graph exists in this repo. Not measured."})
    else:
        rows.append({"id":"onnx","label":"ONNX Runtime / CoreML","backend":"onnx","runnable":False,"e2e_s":None,"notes":"onnxruntime imported but no Kimodo ONNX graph is present."})

def probe_nvidia(rows):
    path=Path("evidence/compare/nvidia-probe.txt")
    notes = path.read_text()[:1200] if path.is_file() else "NVIDIA load probe still pending or missing."
    rows.append({"id":"nvidia-pytorch","label":"NVIDIA Kimodo (PyTorch MPS)","backend":"torch-mps","runnable":False,"e2e_s":None,"notes":notes})

def main():
    rows=[]
    capture_mlx(rows)
    capture_cpp(rows, "metal", "cpp-metal")
    capture_cpp(rows, "cpu", "cpp-cpu")
    probe_onnx(rows)
    probe_nvidia(rows)
    quality={}
    mlx=OUT/"mlx-metal.json"; metal=OUT/"cpp-metal.json"
    if mlx.is_file() and metal.is_file():
        a=np.array(json.loads(mlx.read_text())["posed"])
        b=np.array(json.loads(metal.read_text())["posed"])
        quality={"posed_mae_mlx_vs_cpp_metal": float(np.mean(np.abs(a-b))), "posed_max_mlx_vs_cpp_metal": float(np.max(np.abs(a-b)))}
    report={"prompt":PROMPT.strip(),"frames":FRAMES,"steps":STEPS,"seed":SEED,"fps":FPS,"backends":rows,"quality":quality}
    (OUT/"timings.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: (v if k!="backends" else [{kk:rr.get(kk) for kk in ("id","e2e_s","runnable","encode_s","sample_s")} for rr in v]) for k,v in report.items()}, indent=2))

if __name__=="__main__":
    main()
