"""Simulator-process infra hook (tools/run_episode.py prepends this directory to PYTHONPATH; harness, not controller).

Kit (omni/kit/app/_impl/__init__.py ``_setup_gc``) disables Python's automatic cyclic GC during startup and calls
``gc.enable()`` on the GLOBAL_EVENT_APP_READY event. The first automatic collection after "app ready" can segfault
inside AppLauncher.__init__, before runner/bootstrap.py's own wrapper re-disables GC.
The harness policy is already "no automatic cyclic GC for the whole episode" (bootstrap.py gc.disable() + wrapper);
this hook closes the app-ready window by turning gc.enable() into a recorded no-op. Explicit gc.collect() calls
(lifecycle_gc teardown) are unchanged. Every blocked call is printed to process.log.
"""
import gc as _gc
import sys as _sys

_blocked = []


def _ar_gc_enable_blocked():
    _blocked.append(1)
    try:
        caller = _sys._getframe(1)
        where = f'{caller.f_code.co_filename}:{caller.f_lineno}'
    except Exception:
        where = '?'
    print(f'[ar-simsite] gc.enable() blocked (#{len(_blocked)}) from {where}; automatic GC stays disabled',
          file=_sys.stderr, flush=True)


_gc.enable = _ar_gc_enable_blocked
_gc.disable()

# Periodic visual trace (tools/simsite/ar_video_trace.py): only when run_episode sets AR_VIDEO_TRACE_DIR.
if __import__('os').environ.get('AR_VIDEO_TRACE_DIR'):
    try:
        import ar_video_trace as _ar_video_trace
        _ar_video_trace.install()
    except Exception as _exc:  # never break the simulator start for a debugging aid
        print(f'[ar-simsite] video trace hook not installed: {_exc!r}', file=_sys.stderr, flush=True)
