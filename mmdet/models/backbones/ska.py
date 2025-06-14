import torch
from torch.autograd import Function
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd
import math

def _grid(numel: int, bs: int) -> tuple:
    return (triton.cdiv(numel, bs),)

@triton.jit
def _idx(i, n: int, c: int, h: int, w: int):
    ni = i // (c * h * w)
    ci = (i // (h * w)) % c
    hi = (i // w) % h
    wi = i % w
    m = i < (n * c * h * w)
    return ni, ci, hi, wi, m

@triton.jit
def ska_fwd(
    x_ptr, w_ptr, o_ptr,
    n: int, ic: int, h: int, w: int, ks: int, pad: int, wc: int,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            x_off = ((ni * ic + ci) * h + hin) * w + win
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

    tl.store(o_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_x(
    go_ptr, w_ptr, gi_ptr,
    n: int, ic: int, h: int, w: int, ks: int, pad: int, wc: int,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kh in range(ks):
        ho = hi + pad - kh
        hb = (ho >= 0) & (ho < h)
        for kw in range(ks):
            wo = wi + pad - kw
            b = hb & (wo >= 0) & (wo < w)

            go_off = ((ni * ic + ci) * h + ho) * w + wo
            w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

            go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
            val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

    tl.store(gi_ptr + offs, val.to(CT), mask=m)

@triton.jit
def ska_bwd_w(
    go_ptr, x_ptr, gw_ptr,
    n: int, wc: int, h: int, w: int, ic: int, ks: int, pad: int,
    BS: tl.constexpr,
    CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)
            w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

            val = tl.zeros((BS,), dtype=AT)
            steps = (ic - ci + wc - 1) // wc
            for s in range(tl.max(steps, axis=0)):
                cc = ci + s * wc
                cm = (cc < ic) & m & b

                x_off = ((ni * ic + cc) * h + hin) * w + win
                go_off = ((ni * ic + cc) * h + hi) * w + wi

                x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                val += tl.where(cm, x_val * go_val, 0.0).to(AT)

            tl.store(gw_ptr + w_off, val.to(CT), mask=m)

class SkaFn(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        ctx.ks, ctx.pad = ks, pad
        n, ic, h, width = x.shape
        wc = w.shape[1]
        o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
        numel = o.numel()

        x = x.contiguous()
        w = w.contiguous()

        grid = lambda meta: _grid(numel, meta["BS"])

        ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
        at = tl.float32 if x.dtype == torch.float16 else ct

        ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

        ctx.save_for_backward(x, w)
        ctx.ct, ctx.at = ct, at
        return o

    @staticmethod
    @custom_bwd
    def backward(ctx, go: torch.Tensor) -> tuple:
        ks, pad = ctx.ks, ctx.pad
        x, w = ctx.saved_tensors
        n, ic, h, width = x.shape
        wc = w.shape[1]

        go = go.contiguous()
        gx = gw = None
        ct, at = ctx.ct, ctx.at

        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            numel = gx.numel()
            ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

        if ctx.needs_input_grad[1]:
            gw = torch.empty_like(w)
            numel = gw.numel() // w.shape[2]
            ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)

        return gx, gw, None, None

class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Check if we're in JIT tracing mode
        if torch.jit.is_tracing():
            # Use pure PyTorch implementation for FLOPs calculation
            return self._pytorch_forward(x, w)
        return SkaFn.apply(x, w) # type: ignore
    
    def _pytorch_forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation for compatibility with FLOPs calculation"""
        n, ic, h, width = x.shape
        wc = w.shape[1]
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        
        # Unfold input for convolution
        x_unfold = torch.nn.functional.unfold(
            x, kernel_size=ks, padding=pad, stride=1
        )  # [n, ic*ks*ks, h*w]
        
        # Reshape weight: [n, wc, ks*ks, h, w] -> [n, wc, ks*ks*h*w]
        w_reshaped = w.view(n, wc, -1)
        
        # Prepare output
        output = torch.zeros(n, ic, h * width, device=x.device, dtype=x.dtype)
        
        # Compute convolution for each channel group
        for i in range(0, ic, wc):
            end_c = min(i + wc, ic)
            c_slice = slice(i, end_c)
            
            # Extract relevant channels from unfolded input
            start_idx = i * ks * ks
            end_idx = end_c * ks * ks
            x_slice = x_unfold[:, start_idx:end_idx, :]  # [n, (end_c-i)*ks*ks, h*w]
            
            # Get corresponding weights
            w_idx = slice(0, end_c - i)
            w_slice = w_reshaped[:, w_idx, :]  # [n, (end_c-i), ks*ks*h*w]
            
            # Reshape for batch matrix multiplication
            x_slice = x_slice.view(n, end_c - i, ks * ks, h * width)  # [n, c, ks*ks, h*w]
            w_slice = w_slice.view(n, end_c - i, ks * ks, h * width)  # [n, c, ks*ks, h*w]
            
            # Element-wise multiplication and sum over kernel dimension
            conv_result = (x_slice * w_slice).sum(dim=2)  # [n, c, h*w]
            
            # Store result
            output[:, c_slice, :] = conv_result
        
        return output.view(n, ic, h, width)
