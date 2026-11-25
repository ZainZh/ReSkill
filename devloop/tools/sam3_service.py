#!/usr/bin/env python3
"""Shared SAM 3 run-time service (local perception for the task programs; not part of any bundle). Same port 18891 on
every simulation host, bound to 127.0.0.1.

Model: facebookresearch/sam3 main@7f473cfd + sam3.pt (sha256 prefix 9999e234), bf16 autocast, one model behind one
lock (serial inference), per-process GPU memory cap --mem-gb (default 10 GB, torch set_per_process_memory_fraction).
Python environment: $RESKILL_ROOT/envs/localvision (py3.12, torch 2.7.1+cu126).
Model files and the frozen grounding bundle are looked up in AR_SAM3_EVAL (default $RESKILL_ROOT/eval_sam3_grounding):
third_party/sam3_src, third_party/sam3_weights/sam3.pt and frozen_v2/ (bank.py, ground_v2.py, calib_v2.json,
index.json, SHA256SUMS). frozen_v2 is an offline-calibrated grounding bundle that is NOT released with this code; the
service verifies it at start-up, so without it only a modified copy that skips /detect and /ground can serve /segment.

POST /detect   raw text-prompted SAM 3:
  {image_png: b64, prompts: [{phrase, src?, crop?: [x0,y0,x1,y1], boxes?: [[x0,y0,x1,y1],..], exemplars?: bool}],
   keep?: 10, min_score?: .02, exemplars?: [{image_png: b64, box_xyxy: [..]}]}
  -> {dets: [{phrase, src, score, prob, presence, box, mbox, px, bits}], timing}
     score = prob x presence; box is the SAM 3 predicted box, mbox the mask bounding box (both in full-frame pixels),
     bits = packbits of the mask inside mbox (base64, row-major).
     crop: crop first, then infer (enlarges small targets); results are mapped back to full-frame coordinates.
     boxes: positive box prompts in image (or crop) coordinates.
     exemplars=true: the frozen_v2 "tiled exemplar" scheme: request-level exemplars are pasted in a 160 px strip
     below the frame, their boxes used as positive box prompts; only detections inside the frame are kept.
POST /ground   v2 grounding (frozen_v2, SHA256SUMS verified at start-up; variant v2_full_cal: phrase bank + tiling +
  exemplars + attribute code + exclusive assignment + depth prior from the simulator depth image + calibrated
  thresholds):
  {task, image_png, depth_npy (np.save bytes b64, float32 metres), K, T_world_camera (USD), instruction, queries?}
  -> {queries: {qid: {detections: [{score, box_xyxy, mask_packbits, shape, pixels, phrase, n_phrases, src}], qfeat}}, timing}
  Without queries, the task's offline-evaluation query table in tools/sam3_v2_queries.json is used.
POST /segment  single-box interactive instance segmentation (SAM 3's built-in SAM1-style inst_interactive_predictor),
  same protocol as SAM 2 interactive segmentation:
  {image_sha256, image_png: b64, box_xyxy: [x0,y0,x1,y1] | points_xy+point_labels} -> {image_sha256, shape, mask_packbits, score, provenance}
  Selection rule: multimask_output=False, the (single) mask with the highest predicted IoU,
  threshold 0 (logit), no post-processing; score = predicted IoU.
GET / or /health -> service, host, GPU, weight/frozen-file sha, statistics, GPU memory.
Reads no ground truth; only processes the observation images given in the request.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# tools/queue.py shadows the stdlib queue (torch imports it): move the script directory to the end of sys.path
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _HERE] + [_HERE]

EVAL = Path(os.environ.get('AR_SAM3_EVAL') or Path(__file__).resolve().parents[2] / 'eval_sam3_grounding')
SRC = EVAL / 'third_party' / 'sam3_src'
CKPT = EVAL / 'third_party' / 'sam3_weights' / 'sam3.pt'
CKPT_SHA = '9999e234'           # prefix; full value in third_party/sam3_weights/SHA256.txt
FROZEN = EVAL / 'frozen_v2'
QUERIES_FILE = Path(__file__).resolve().parent / 'sam3_v2_queries.json'
TILES = [(x0, y0, x0 + 256, y0 + 180) for y0 in (40, 180) for x0 in (0, 192, 384)]
STRIP = 160
VARIANT = 'v2_full_cal'
FIG_KIND = {'cactus in pot': 'cactus in pot', 'iron man figure': 'iron man', 'hulk figure': 'hulk'}


def verify_frozen():
    sums = {}
    for line in (FROZEN / 'SHA256SUMS').read_text().split('\n'):
        if line.strip():
            h, name = line.split()
            if hashlib.sha256((EVAL / name).read_bytes()).hexdigest() != h:
                raise SystemExit(f'frozen file changed: {name}')
            sums[name] = h
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=18891)
    ap.add_argument('--mem-gb', type=float, default=10.0)
    a = ap.parse_args()
    if a.port in (18873, 18895, 18897):
        raise SystemExit(f'port {a.port} belongs to another service')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    sums = verify_frozen()
    sys.path.insert(0, str(FROZEN))
    sys.path.append(str(SRC))
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model import box_ops
    import bank
    import ground_v2

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2**30 / total))
    t0 = time.time()
    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False, device='cuda',
                                   enable_inst_interactivity=True)
    proc = Sam3Processor(model, confidence_threshold=0.0)
    load_s = time.time() - t0
    calib = json.loads((FROZEN / 'calib_v2.json').read_text())
    exidx = json.loads((FROZEN / 'index.json').read_text())
    qtable = json.loads(QUERIES_FILE.read_text())
    lock = threading.Lock()
    stats = dict(detect=0, ground=0, segment=0, errors=0, busy_s=0.0, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    # ---------------- inference core (= run/text_runs of the offline evaluation's run_sam3_bank.py) ----------------
    def run(state, W, H, offset, clip, keep, min_score):
        out = model.forward_grounding(backbone_out=state['backbone_out'], find_input=proc.find_stage,
                                      geometric_prompt=state['geometric_prompt'], find_target=None)
        prob = out['pred_logits'].sigmoid().squeeze(-1)[0].float()
        pres = out['presence_logit_dec'].sigmoid().float().reshape(-1)[0]
        score = prob * pres
        order = torch.argsort(score, descending=True)[:keep * 2]
        order = order[score[order] >= min_score]
        dets = []
        if len(order):
            boxes = (box_ops.box_cxcywh_to_xyxy(out['pred_boxes'][0][order].float())
                     * torch.tensor([W, H, W, H], device='cuda')).cpu().numpy()
            masks = (F.interpolate(out['pred_masks'][0][order].unsqueeze(1).float(), (H, W), mode='bilinear',
                                   align_corners=False)[:, 0] > 0).cpu().numpy()
            for j, i in enumerate(order.tolist()):
                m, b = masks[j], boxes[j]
                if clip is not None:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if not (clip[0] <= cx < clip[2] and clip[1] <= cy < clip[3]):
                        continue
                    m = m[clip[1]:clip[3], clip[0]:clip[2]]
                    b = [max(b[0], 0), max(b[1], 0), min(b[2], clip[2]), min(b[3], clip[3])]
                ys, xs = np.nonzero(m)
                if not len(xs):
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                sub = m[y0:y1, x0:x1]
                ox, oy = offset
                dets.append(dict(score=float(score[i]), prob=float(prob[i]), presence=float(pres),
                                 box=[float(b[0]) + ox, float(b[1]) + oy, float(b[2]) + ox, float(b[3]) + oy],
                                 mbox=[x0 + ox, y0 + oy, x1 + ox, y1 + oy], px=int(sub.sum()),
                                 bits=base64.b64encode(np.packbits(sub.reshape(-1))).decode()))
                if len(dets) >= keep:
                    break
        return float(pres), dets

    def text_runs(img, phrases, src, offset=(0, 0), clip=None, boxes=None, keep=10, min_score=.02, state=None):
        """boxes: normalised cxcywh positive box prompts (SAM3 geometric prompt)."""
        W, H = img.size
        st = state or proc.set_image(img)
        res = []
        for ph in phrases:
            proc.reset_all_prompts(st)
            t = time.time()
            st['backbone_out'].update(model.backbone.forward_text([ph], device='cuda'))
            st['geometric_prompt'] = model._get_dummy_prompt()
            for bx in boxes or []:
                st['geometric_prompt'].append_boxes(torch.tensor(bx, device='cuda', dtype=torch.float32).view(1, 1, 4),
                                                    torch.tensor([True], device='cuda').view(1, 1))
            pres, dets = run(st, W, H, offset, clip, keep, min_score)
            res.append(dict(src=src, phrase=ph, presence=pres, prompt_s=time.time() - t, dets=dets))
        return res

    def norm_boxes(boxes, W, H):
        return [[(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H] for b in boxes]

    def composite(img, exemplars):
        """frozen_v2 tiled exemplars: frame on top, exemplar strip (each <= 160x160) below; returns (tile, exemplar boxes cxcywh normalised)."""
        W, H = img.size
        comp = Image.new('RGB', (W, H + STRIP), (0, 0, 0))
        comp.paste(img, (0, 0))
        boxes = []
        for j, (ex, box) in enumerate(exemplars[:W // STRIP]):
            s = min(STRIP / ex.width, STRIP / ex.height)
            ex2 = ex.resize((max(1, int(ex.width * s)), max(1, int(ex.height * s))))
            x0, y0 = j * STRIP, H
            comp.paste(ex2, (x0, y0))
            bx = [v * s for v in box]
            boxes.append([x0 + bx[0], y0 + bx[1], x0 + bx[2], y0 + bx[3]])
        return comp, norm_boxes(boxes, W, H + STRIP)

    def detect(req):
        img = Image.open(io.BytesIO(base64.b64decode(req['image_png']))).convert('RGB')
        keep, min_score = int(req.get('keep', 10)), float(req.get('min_score', .02))
        exs = [(Image.open(io.BytesIO(base64.b64decode(e['image_png']))).convert('RGB'), [float(v) for v in e['box_xyxy']])
               for e in req.get('exemplars') or []]
        groups = {}
        for p in req['prompts']:
            crop = tuple(int(v) for v in p['crop']) if p.get('crop') else None
            boxes = tuple(tuple(float(v) for v in b) for b in p.get('boxes') or ())
            key = (p.get('src', 'full'), crop, bool(p.get('exemplars')) and bool(exs), boxes)
            groups.setdefault(key, []).append(p['phrase'])
        out, t = [], time.time()
        with lock:
            tw = time.time()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                states = {}
                for (src, crop, use_ex, boxes), phrases in groups.items():
                    im = img.crop(crop) if crop else img
                    off = crop[:2] if crop else (0, 0)
                    if use_ex:
                        comp, exb = composite(im, exs)
                        runs = text_runs(comp, phrases, src, off, (0, 0, im.size[0], im.size[1]),
                                         exb + norm_boxes(boxes, *comp.size), keep, min_score)
                    else:
                        sk = crop                      # encode each crop only once
                        if sk not in states:
                            states[sk] = proc.set_image(im)
                        runs = text_runs(im, phrases, src, off, None, norm_boxes(boxes, *im.size), keep, min_score,
                                         state=states[sk])
                    for r in runs:
                        out += [dict(d, phrase=r['phrase'], src=src) for d in r['dets']]
                torch.cuda.synchronize()
            busy = time.time() - tw
        stats['detect'] += 1
        stats['busy_s'] += busy
        return dict(dets=out, timing=dict(wait_s=round(tw - t, 3), total_s=round(time.time() - t, 3),
                                          prompts=len(req['prompts']), encodes=len(groups)))

    def bank_record(img, task, instruction):
        rec = dict(task=task, tag='runtime', runs=[])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            rec['runs'] += text_runs(img, bank.frame_phrases(task, instruction), 'full')
            tp = bank.tile_phrases(task, instruction)
            for i, tb in enumerate(TILES if tp else []):
                rec['runs'] += text_runs(img.crop(tb), tp, f'tile{i}', offset=tb[:2])
            for label, phrase in bank.EXEMPLAR.get(task, {}).items():
                exs = exidx.get(f'{task}|{label}', [])[:4]
                if not exs:
                    continue
                comp, boxes = composite(img, [(Image.open(e['file']).convert('RGB'), e['box_in_crop']) for e in exs])
                rec['runs'] += text_runs(comp, [phrase], f'ex_text:{label}', clip=(0, 0, 640, 480), boxes=boxes)
                rec['runs'] += text_runs(comp, ['visual'], f'ex_only:{label}', clip=(0, 0, 640, 480), boxes=boxes)
        return rec

    def ground(req):
        task = req['task']
        queries = req.get('queries') or qtable.get(task)
        if not queries:
            raise ValueError(f'no query list for task {task}')
        png = base64.b64decode(req['image_png'])
        img = Image.open(io.BytesIO(png)).convert('RGB')
        if img.size != (640, 480):
            raise ValueError(f'expected 640x480, got {img.size}')
        depth = np.load(io.BytesIO(base64.b64decode(req['depth_npy'])))
        t0 = time.time()
        with lock:
            tw = time.time()
            rec = bank_record(img, task, req.get('instruction', ''))
            tb = time.time()
        with tempfile.TemporaryDirectory(dir='/dev/shm' if os.path.isdir('/dev/shm') else None) as td:
            td = Path(td)
            (td / 'head_rgb.png').write_bytes(png)
            np.save(td / 'depth.npy', depth.astype(np.float32))
            (td / 'meta.json').write_text(json.dumps(dict(task=task, instruction=req.get('instruction', ''), K=req['K'],
                                                          T_world_camera=req['T_world_camera'])))
            sc = ground_v2.Scene(td)
            opt = dict(ground_v2.VARIANTS[VARIANT], name=VARIANT)
            res = ground_v2.ground_frame(task, 'runtime', queries, opt, calib, rec, sc)
        out = {r['qid']: dict(detections=r.get('detections', []), qfeat=r.get('qfeat')) for r in res}
        attrs = {}
        if task == 'organize_table' and out.get('figurine', {}).get('detections'):
            d = out['figurine']['detections'][0]
            fm = np.unpackbits(np.frombuffer(base64.b64decode(d['mask_packbits']), np.uint8))[:480 * 640].reshape(480, 640).astype(bool)
            best, kind = 0., 'other'
            for r in rec['runs']:
                if r['src'] == 'full' and r['phrase'] in FIG_KIND:
                    for dd in r['dets']:
                        m = ground_v2.decode(dd)
                        iou = (m & fm).sum() / max(1, (m | fm).sum())
                        if iou >= .5 and dd['score'] > best:
                            best, kind = dd['score'], FIG_KIND[r['phrase']]
            attrs['figurine'] = dict(kind=kind, kind_score=best)
        stats['ground'] += 1
        stats['busy_s'] += tb - tw
        return dict(queries=out, attributes=attrs, variant=VARIANT,
                    timing=dict(wait_s=tw - t0, bank_s=tb - tw, total_s=time.time() - t0, runs=len(rec['runs'])))

    seg_prov = dict(backend='sam3_inst_interactive', model='facebook/sam3 image + inst_interactive_predictor',
                    checkpoint=str(CKPT), checkpoint_sha256_prefix=CKPT_SHA, service_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest(), endpoint=f'http://127.0.0.1:{a.port}/segment',
                    multimask_output=False, training='Generic pretrained SAM3; no task training or task scene assets')

    def segment(req):
        raw = base64.b64decode(req['image_png'])
        if hashlib.sha256(raw).hexdigest() != req['image_sha256']:
            raise ValueError('Image hash mismatch')
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        box, points, labels = req.get('box_xyxy'), req.get('points_xy'), req.get('point_labels')
        if box is None and points is None:
            raise ValueError('SAM requires RGB-derived semantic box or points')
        with lock:
            tw = time.time()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                st = proc.set_image(img)
                masks, scores, _ = model.predict_inst(
                    st, box=None if box is None else np.asarray(box, np.float32),
                    point_coords=None if points is None else np.asarray(points, np.float32),
                    point_labels=None if labels is None else np.asarray(labels, np.int32),
                    multimask_output=False)
            stats['busy_s'] += time.time() - tw
        stats['segment'] += 1
        i = int(np.argmax(scores))
        mask = np.asarray(masks[i], bool)
        return dict(mask_packbits=base64.b64encode(np.packbits(mask).tobytes()).decode(), shape=list(mask.shape),
                    score=float(scores[i]), image_sha256=req['image_sha256'], provenance=seg_prov)

    def health():
        return dict(ok=True, service='sam3_shared', model='facebook/sam3 image', checkpoint_sha256_prefix=CKPT_SHA,
                    variant=VARIANT, frozen_sha256=sums, host=socket.gethostname(), gpu=gpu, port=a.port, pid=os.getpid(),
                    load_s=round(load_s, 1), stats=stats, mem_cap_gb=a.mem_gb,
                    mem_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    tasks_with_queries=sorted(qtable))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/health'):
                self._send(200, health())
            else:
                self._send(404, dict(error='not found'))

        def do_POST(self):
            fn = {'/detect': detect, '/ground': ground, '/segment': segment}.get(self.path)
            if fn is None:
                return self._send(404, dict(error='unknown path'))
            t = time.time()
            try:
                req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                resp = fn(req)
                print(time.strftime('%H:%M:%S'), self.path, 'ok', f"{time.time() - t:.2f}s", flush=True)
                self._send(200, resp)
            except Exception as exc:    # noqa: BLE001
                stats['errors'] += 1
                print(time.strftime('%H:%M:%S'), self.path, 'error', repr(exc)[:300], f'{time.time() - t:.2f}s', flush=True)
                self._send(500, dict(error=repr(exc)[:500]))

    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    print(f'sam3 shared service on 127.0.0.1:{a.port} gpu {gpu} load {load_s:.1f}s frozen {sums}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
