"""One prefill forward per agent turn on GLM-5.3's hybrid layers (GLM53_RC_PAGE=1).

exllamav3 1.5.1 resumes a cached prompt from a snapshot of the recurrent (KDA, linear-attention) state taken at a page
boundary. It takes those snapshots every `recurrent_checkpoint_interval` tokens (2048 for GLM-5.3) and at the last page
of each prompt, which it gets by splitting the prompt's prefill into two forwards ("a separate forward pass for the last
page", Job.prefill). A follow-up turn restores the latest snapshot before the point where its prompt differs from what
the cache holds and prefills the rest.

With the routed experts in host RAM, every prefill forward of more than a few tokens reads nearly the whole non-cached
expert bank over PCIe, whatever its token count (PCIe 3.0, 1 user: +64 tokens 4.8 s, +256 tokens 8.2 s, the link
saturated throughout). An agent turn (previous reply + tool result) therefore paid for two forwards, ~16 s before its
first token.

Mode "page": snapshot at every page boundary during decode (interval 256; ~145 MB of state per snapshot for
GLM-5.3-Flash, tens of ms to copy) and drop the last-page split. The next turn restores within 255 tokens of where its
prompt diverges and prefills in one forward. Decode lookahead (k_lookahead) never runs across a page boundary, so every
snapshot matches its position. Mode "stock" is exllamav3's own behaviour. Switch at runtime with set_mode() (serve.py
GET /rc?mode=page|stock), only while no job runs.

Cost: a turn whose prompt differs right after the previous prompt's end (a short user message after a reply that was cut
short) restores from further back. More snapshots also need a larger snapshot cache: serve with -rcs 24 (GB; the
default 4 GB holds ~27).

Env: GLM53_RC_PAGE=1 (install; starts in page mode), GLM53_RC_PAGE_ON=0 (install but start in stock mode).
"""
import inspect
import textwrap

PAGE_INTERVAL = 256
MODE = "stock"
STATS = {"switches": 0}
_GENS = []
_installed = False


def install():
    """Patch exllamav3 1.5.1's Job.prefill and Generator.__init__. Call before the generator is created."""
    global _installed
    if _installed:
        return
    from exllamav3.version import __version__
    import exllamav3.generator.generator as xgen
    import exllamav3.generator.job as xjob
    assert __version__ == "1.5.1", f"k_rcpage is written for exllamav3 1.5.1, not {__version__}"
    split = "if prefill_start < last_page_b <= prefill_end:"
    src = textwrap.dedent(inspect.getsource(xjob.Job.prefill))
    assert src.count(split) == 1, "k_rcpage: exllamav3's last-page split was not found exactly once"
    xjob._RC_SPLIT = [True]
    patched = {}
    exec(compile(src.replace(split, "if _RC_SPLIT[0] and " + split[3:]), xjob.__file__, "exec"), vars(xjob), patched)
    xjob.Job.prefill = patched["prefill"]
    init = xgen.Generator.__init__

    def generator_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._rc_stock_interval = self.recurrent_checkpoint_interval
        _GENS.append(self)
        _apply()

    xgen.Generator.__init__ = generator_init
    _installed = True
    _apply()
    print(f" -- k_rcpage: installed (mode {MODE})", flush=True)


def _apply():
    import exllamav3.generator.job as xjob
    xjob._RC_SPLIT[0] = MODE != "page"
    for g in _GENS:
        g.recurrent_checkpoint_interval = PAGE_INTERVAL if MODE == "page" else g._rc_stock_interval


def set_mode(mode):
    global MODE
    assert mode in ("page", "stock"), mode
    if mode != MODE:
        MODE = mode
        STATS["switches"] += 1
        if _installed:
            _apply()
    return MODE


def summary():
    return dict(STATS, mode=MODE, installed=_installed)
