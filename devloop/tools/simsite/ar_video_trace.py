"""Harness-side periodic visual trace of a native episode (simulator process only; NOT part of any bundle).

Why: the native eval client (src/eval_client/eval_env.py) appends a frame to its episode mp4 only inside
``get_obs_batch`` (``_stream_vision``), i.e. only when somebody calls ``env.get_obs()``. A native policy does that
every action; our graph controller acts through ``take_action`` and reads the cameras only when it observes
(a handful of times), so eval_result/**/episode_*_cam_*.mp4 held only the first few frames + the terminal frame.

What: installed by tools/simsite/sitecustomize.py as an import hook on ``src.eval_client.eval_env``. It wraps
``create_eval_env`` and, on the returned EvalEnv class, ``take_action_batch``. Every ``AR_VIDEO_EVERY`` native
actions (default 10; 0 = off) and once when ``end_flag`` first becomes true it
  1. renders the capture products and reads the native observation (obs_manager.render_for_capture/get_obs —
     the same calls the native get_obs_batch makes, without its PhysX checks or deepcopy) and appends the
     frame to the native video streams (``_stream_vision``), so the native mp4s become a usable trace;
  2. writes ``$AR_VIDEO_TRACE_DIR/f_<actions:05d>.jpg``: head camera 640x480 + both wrist cameras 320x240
     stacked to its right, with the action count stamped, and one line per sample in ``index.jsonl``.
run_episode.py stitches the JPEGs into ``video_trace/trace.mp4`` after the process exits.
Rendering does not step physics; the controller never sees these frames. Any error is logged to stderr,
counted, and after 3 errors the hook disables itself; it never raises into the episode.
"""
import json
import os
import sys
import time

EVERY = int(os.environ.get('AR_VIDEO_EVERY', '10') or 0)
OUT = os.environ.get('AR_VIDEO_TRACE_DIR')
TARGET = 'src.eval_client.eval_env'
_state = dict(errors=0, samples=0, seconds=0.0, disabled=False, last_cnt=None, end_done=False)


def _log(msg):
    print(f'[ar-video-trace] {msg}', file=sys.stderr, flush=True)


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    import numpy as np
    return np.asarray(value.numpy() if hasattr(value, 'numpy') else value)


def _compose(vision, cnt):
    import cv2
    import numpy as np

    def color(key):
        data = vision.get(key)
        img = data.get('color') if isinstance(data, dict) else None
        if img is None:
            return None
        img = _np(img)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            return None
        img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img * (255. if img.max() <= 1.0 else 1.), 0, 255).astype(np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)

    head = color('cam_head')
    others = [k for k in vision if k != 'cam_head']
    if head is None and others:
        head = color(others.pop(0))
    canvas = np.zeros((480, 960, 3), np.uint8)
    if head is not None:
        canvas[:, :640] = cv2.resize(head, (640, 480), interpolation=cv2.INTER_AREA)
    order = [k for k in ('cam_left_wrist', 'cam_right_wrist') if k in others] + \
            [k for k in others if k not in ('cam_left_wrist', 'cam_right_wrist')]
    for i, key in enumerate(order[:2]):
        img = color(key)
        if img is not None:
            canvas[i * 240:(i + 1) * 240, 640:] = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, key.replace('cam_', ''), (646, i * 240 + 18), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'action {cnt}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f'action {cnt}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _sample(env, cnt, end):
    t0 = time.monotonic()
    om = env.obs_manager
    om.render_for_capture()
    data = om.get_obs(env_idx_list=[0])[0]
    if not end:                                     # native stream only while running (as get_obs_batch does)
        env._stream_vision(0, data)
    row = dict(actions=cnt, end_flag=bool(end), epoch=time.time())
    if OUT:
        import cv2
        os.makedirs(OUT, exist_ok=True)
        name = f'f_{cnt:05d}{"_end" if end else ""}.jpg'
        cv2.imwrite(os.path.join(OUT, name), _compose(data.get('vision') or {}, cnt), [cv2.IMWRITE_JPEG_QUALITY, 80])
        row['file'] = name
    row['seconds'] = round(time.monotonic() - t0, 4)
    _state['samples'] += 1
    _state['seconds'] += row['seconds']
    if OUT:
        with open(os.path.join(OUT, 'index.jsonl'), 'a') as f:
            f.write(json.dumps(row) + '\n')


def _after_actions(env):
    if _state['disabled']:
        return
    try:
        cnt = int(env.take_action_cnt[0])
        end = bool(env.end_flag[0])
        if end:
            if _state['end_done']:
                return
            _state['end_done'] = True
        elif cnt == _state['last_cnt'] or cnt % EVERY:
            return
        _state['last_cnt'] = cnt
        _sample(env, cnt, end)
    except Exception as exc:                        # never propagate into the episode
        _state['errors'] += 1
        _log(f'sample failed ({_state["errors"]}): {type(exc).__name__}: {exc}')
        if _state['errors'] >= 3:
            _state['disabled'] = True
            _log('disabled after 3 errors')


def _patch_class(cls):
    if getattr(cls, '_ar_video_trace', False):
        return
    original = cls.take_action_batch

    def take_action_batch(self, *a, **kw):
        result = original(self, *a, **kw)
        _after_actions(self)
        return result

    cls.take_action_batch = take_action_batch
    cls._ar_video_trace = True
    _log(f'installed on {cls.__name__}.take_action_batch every={EVERY} out={OUT}')


def _patch_module(module):
    original = module.create_eval_env

    def create_eval_env(*a, **kw):
        env = original(*a, **kw)
        try:
            _patch_class(type(env))
        except Exception as exc:
            _log(f'install failed: {type(exc).__name__}: {exc}')
        return env

    module.create_eval_env = create_eval_env


class _Finder:
    """Meta-path finder: patch ``src.eval_client.eval_env`` right after it executes (before main.py binds it)."""

    def find_spec(self, name, path=None, target=None):
        if name != TARGET:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        loader, exec_module = spec.loader, spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            try:
                _patch_module(module)
            except Exception as exc:
                _log(f'module patch failed: {type(exc).__name__}: {exc}')
        loader.exec_module = patched_exec
        return spec


def install():
    if EVERY <= 0:
        return
    sys.meta_path.insert(0, _Finder())
