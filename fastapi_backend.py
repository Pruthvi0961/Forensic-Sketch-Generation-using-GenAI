"""
Forensic Sketch Generation Using GenAI — FastAPI Backend
Wraps SDXL generation + FAISS/BiSeNet biometric matching into a REST API.
Run this inside Google Colab after running the setup cell.

Endpoints:
  GET  /health                     — ping
  POST /api/generate               — generate suspect composite from prompt
  POST /api/search                 — run hybrid forensic search on an image
  POST /api/generate_and_search    — full pipeline: generate → search
  POST /api/build_index            — (re)build FAISS index from a folder path
  GET  /api/image/{filename}       — serve generated/match images
"""

import os, sys, io, uuid, shutil, base64, traceback
import cv2, numpy as np, faiss, torch
from PIL import Image
from typing import Optional, List
from torchvision.transforms.functional import to_tensor

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Forensic Sketch Generation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ngrok public URL will call this
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model handles (set by load_models()) ────────────────────────────
pipe   = None     # SDXL pipeline
fa_app = None     # InsightFace
net    = None     # BiSeNet

# FAISS index + metadata
index        = None
db_embeddings = []
db_metadata   = []

# Working dirs
OUTPUT_DIR = "/content/criminet_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
def load_models():
    global pipe, fa_app, net
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from insightface.app import FaceAnalysis
    sys.path.append('/content/face-parsing.PyTorch')
    from model import BiSeNet

    print(">> Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    print(">> Loading InsightFace...")
    fa_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    fa_app.prepare(ctx_id=0, det_size=(640, 640))

    print(">> Loading BiSeNet...")
    net = BiSeNet(n_classes=19)
    ckpt = '/content/face-parsing.PyTorch/res/cp/79999_iter.pth'
    net.cuda()
    net.load_state_dict(torch.load(ckpt))
    net.eval()

    print("✅ All models loaded.")

# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────
def get_mask_color(img_rgb, mask, label_id):
    pixels = img_rgb[mask == label_id]
    if len(pixels) < 10: return None
    return np.mean(pixels, axis=0).astype(int).tolist()

def get_mask_color_hsv(img_bgr, mask, label_id):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == label_id]
    if len(pixels) < 10: return None
    return np.mean(pixels[:, :2], axis=0).astype(int).tolist()

def get_hsv_similarity(hsv1, hsv2):
    if hsv1 is None or hsv2 is None: return 0.5
    dst = np.linalg.norm(np.array(hsv1) - np.array(hsv2))
    return max(0.0, 1 - (dst / 312.0))

def parse_face(img_bgr):
    """Run BiSeNet on an image, return (mask, skin_hsv, hair_rgb, has_beard)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    inp = to_tensor(img_resized).unsqueeze(0).cuda()
    with torch.no_grad():
        out = net(inp)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    mask = cv2.resize(
        parsing.astype(np.uint8),
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    skin_hsv  = get_mask_color_hsv(img_bgr, mask, 1)
    hair_rgb  = get_mask_color(img_rgb, mask, 17)
    has_beard = bool(np.sum(mask == 11) > 500)
    return mask, skin_hsv, hair_rgb, has_beard

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─── REQUEST MODELS ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 35
    guidance: Optional[float] = 5.5

class BuildIndexRequest(BaseModel):
    folder_path: str

class SearchRequest(BaseModel):
    image_b64: str   # base64-encoded PNG/JPG
    k: Optional[int] = 5

class GenerateAndSearchRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 35
    guidance: Optional[float] = 5.5
    k: Optional[int] = 5

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": pipe is not None,
        "index_size": index.ntotal if index is not None else 0,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(503, "Models not loaded. Run load_models() first.")
    try:
        forensic_prompt = (
            f"Mugshot portrait of a suspect: {req.prompt}. "
            "Front-facing view, neutral expression, looking at camera. "
            "Plain grey background, harsh flat lighting, raw photography, "
            "hyper-realistic, 8k uhd, highly detailed skin pores, police photography style."
        )
        neg = req.negative_prompt or (
            "cgi, 3d, render, cartoon, anime, illustration, painting, blurry, "
            "smile, side view, artistic lighting, hat, sunglasses, glowing, "
            "stylized, deformed, low quality, bad eyes, fake skin"
        )
        image = pipe(
            prompt=forensic_prompt,
            negative_prompt=neg,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            width=832, height=1216
        ).images[0]

        fname = f"suspect_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        image.save(fpath)

        return {
            "success": True,
            "filename": fname,
            "image_b64": img_to_b64(fpath),
            "prompt_used": forensic_prompt,
        }
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {traceback.format_exc()}")


@app.post("/api/search")
def search(req: SearchRequest):
    global index, db_metadata
    if index is None or len(db_metadata) == 0:
        raise HTTPException(503, "Database index not built. Call /api/build_index first.")
    try:
        img_bytes = base64.b64decode(req.image_b64)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, "Could not decode image.")

        # Biometric embedding
        faces = fa_app.get(img_bgr)
        if not faces:
            raise HTTPException(400, "No face detected in the query image.")
        face_vec = faces[0].normed_embedding
        gender   = "Male" if faces[0].gender == 1 else "Female"
        age      = int(faces[0].age)

        # Semantic parsing
        _, skin_hsv, hair_rgb, has_beard = parse_face(img_bgr)

        # FAISS search
        qvec = face_vec.reshape(1, -1).astype('float32')
        distances, indices = index.search(qvec, min(300, index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            m = db_metadata[idx]
            skin_sim  = get_hsv_similarity(skin_hsv,  m.get('skin_hsv'))
            hair_sim  = get_hsv_similarity(hair_rgb,  m.get('hair_rgb'))
            bio_dist  = float(distances[0][i])

            skin_penalty  = (1 - skin_sim) * 3.0
            beard_penalty = 0.6 if (has_beard and not m.get('has_beard')) else 0.0
            hair_penalty  = (1 - hair_sim)  * 2.0
            gender_penalty = 0.5 if (gender != m.get('gender', gender)) else 0.0
            final_score   = bio_dist + skin_penalty + beard_penalty + hair_penalty + gender_penalty

            results.append({
                "rank":         i + 1,
                "suspect_id":   m.get('suspect_id', f'UNK-{idx:05d}'),
                "final_score":  round(final_score, 3),
                "bio_dist":     round(bio_dist, 3),
                "skin_match":   round(skin_sim * 100, 1),
                "hair_match":   round(hair_sim * 100, 1),
                "gender":       m.get('gender', '—'),
                "local_path":   m.get('local_path', ''),
            })

        results = sorted(results, key=lambda x: x['final_score'])[:req.k]

        # Re-rank 1..k and attach mugshot b64
        enriched = []
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
            path = r.get('local_path', '')
            r['mugshot_b64'] = img_to_b64(path) if path and os.path.exists(path) else None
            enriched.append(r)

        return {
            "success": True,
            "query_gender": gender,
            "query_age": age,
            "matches": enriched,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, traceback.format_exc())


@app.post("/api/generate_and_search")
def generate_and_search(req: GenerateAndSearchRequest):
    """Full pipeline: generate composite → FAISS search → return both."""
    gen_result = generate(GenerateRequest(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        guidance=req.guidance,
    ))

    fpath = os.path.join(OUTPUT_DIR, gen_result["filename"])
    with open(fpath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    search_result = search(SearchRequest(image_b64=b64, k=req.k))

    return {
        "success": True,
        "generated_image_b64": gen_result["image_b64"],
        "generated_filename":  gen_result["filename"],
        "prompt_used":         gen_result["prompt_used"],
        "query_gender":        search_result.get("query_gender"),
        "query_age":           search_result.get("query_age"),
        "matches":             search_result.get("matches", []),
    }


@app.post("/api/build_index")
def build_index(req: BuildIndexRequest):
    """Recursively index all faces in folder_path into FAISS."""
    global index, db_embeddings, db_metadata
    import tqdm as tqdm_module

    if fa_app is None or net is None:
        raise HTTPException(503, "Models not loaded.")

    folder = req.folder_path
    if not os.path.isdir(folder):
        raise HTTPException(400, f"Folder not found: {folder}")

    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.startswith('.'):
                all_files.append(os.path.join(root, f))

    new_embeddings, new_metadata = [], []
    failed = 0

    for path in tqdm_module.tqdm(all_files, desc="Indexing"):
        img = cv2.imread(path)
        if img is None: failed += 1; continue
        faces = fa_app.get(img)
        if not faces: failed += 1; continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            _, skin_hsv, hair_rgb, has_beard = parse_face(img)
        except Exception:
            failed += 1; continue

        gender  = "Male" if faces[0].gender == 1 else "Female"
        suspect_id = os.path.splitext(os.path.basename(path))[0]

        new_embeddings.append(faces[0].normed_embedding)
        new_metadata.append({
            "suspect_id": suspect_id,
            "local_path": path,
            "skin_hsv":   skin_hsv,
            "hair_rgb":   hair_rgb,
            "has_beard":  has_beard,
            "gender":     gender,
        })

    if not new_embeddings:
        raise HTTPException(400, "No valid faces found in the folder.")

    mat = np.array(new_embeddings).astype('float32')
    new_index = faiss.IndexFlatL2(mat.shape[1])
    new_index.add(mat)

    index         = new_index
    db_embeddings = new_embeddings
    db_metadata   = new_metadata

    return {
        "success": True,
        "indexed": len(new_embeddings),
        "failed":  failed,
        "total":   len(all_files),
        "dim":     mat.shape[1],
    }


@app.get("/api/image/{filename}")
def serve_image(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/png")
