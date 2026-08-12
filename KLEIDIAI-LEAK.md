# Unbounded memory growth on Apple Silicon with SME2 (M4/M5)

**Symptom:** with the Background Removal filter enabled on a video capture source,
OBS memory grows linearly at roughly **18 GB/min** and never plateaus. A 64 GB
machine reaches memory exhaustion in about 11 minutes. No recording or streaming
is needed — previewing is enough — and no filter setting has to be touched.

**Affects:** plugin 1.4.1 and 1.4.0, CPU inference device, on Macs that report
`hw.optional.arm.FEAT_SME2` (M4/M5 generation). Machines without SME2 are unlikely
to hit this.

**Cause:** ONNX Runtime's ARM KleidiAI convolution kernels. Every allocation comes
from `ArmKleidiAI::MlasConv`, reached from the plugin's per-frame inference. The
memory is *reachable* — this is not a dropped-pointer leak, so `leaks` will not
find it.

**Fix used here:** rebuild the plugin against Microsoft's **prebuilt** ONNX Runtime
instead of the project's from-source ORT build. The prebuilt library does not
include the KleidiAI float-convolution path. No source change to the plugin or to
ONNX Runtime is required — this is purely a build configuration difference.

---

## Evidence

### Growth is linear, with no plateau

Two fresh OBS processes, filter enabled, previewing, nothing touched:

| clock | RSS |
|---|---:|
| 12:34:17 | 14.68 GB |
| 12:34:28 | 17.80 GB |
| 12:34:38 | 20.48 GB |
| 12:34:48 | 23.57 GB |
| 12:34:58 | 26.70 GB |
| 12:35:08 | 29.65 GB |

Per-interval deltas: 0.284, 0.268, 0.309, 0.313, 0.295 GB/s — flat, so the curve is
a straight line rather than an allocator arena settling. A second process measured
18.4 GB/min. An earlier unmonitored session reached **203.6 GB** before the kernel
fired a jetsam event.

Notably OBS itself is not killed — it sits at priority 100 while the kernel kills
bystander processes, so there is no OBS crash report. The failure presents as
"your system has run out of application memory" with no obvious culprit.

### The memory is reachable

```
Process: 643905 nodes malloced for 11235822 KB
Process: 60 leaks for 8416 total leaked bytes.
```

60 unreachable blocks totalling 8,416 bytes, out of 11.2 GB. Everything is still
referenced. This is why `leaks` is useless here and why the growth never plateaus.

`heap` attributes it:

```
COUNT      BYTES         AVG      CLASS_NAME                              BINARY
89954   11124938896   123673.6   non-object from obs-backgroundremoval    obs-backgroundremoval
   11      87375872  7943261.0   malloc in video_frame_init               libobs
```

11.12 GB of an 11.6 GB footprint. The next contributor is 127× smaller.

### The call site

Launch OBS with `MallocStackLogging=1`, then `malloc_history <pid> -allBySize`.
The shipped binary is stripped, but the release ships a matching dSYM
(`obs-backgroundremoval.dSYM.tar.zst` on the 1.4.1 release), so `atos` resolves it:

```
background_filter_video_tick            (background-filter.cpp:582)
processImageForBackground               (background-filter.cpp:495)
runFilterModelInference                 (ort-session-utils.cpp:163)
Model::runNetworkInference
OrtApis::Run → InferenceSession::Run → ExecuteGraph → ExecuteThePlan
onnxruntime::ExecuteKernel
onnxruntime::Conv<float>::Compute
MlasConv
ArmKleidiAI::MlasConv                   <- 7,798 calls, 10.96 GB
```

One stack accounted for 95.4% of all live memory in the process. The runner-up was
46× smaller.

### Why it is SME2-specific

The KleidiAI kernels compiled into the binary include
`kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c` and friends — SME2
paths. This machine reports:

```
hw.optional.arm.FEAT_SME: 1
hw.optional.arm.FEAT_SME2: 1
hw.optional.arm.FEAT_SME2p1: 1
```

MLAS selects kernels from these CPU feature bits at runtime, and there is no
environment variable to override the choice — KleidiAI is compiled in via
CMake FetchContent.

### Why it does not reproduce against prebuilt ONNX Runtime

Three standalone reproducers were built against Microsoft's prebuilt ORT 1.23.2 —
one driving `Session::Run` with the same model, session options and pre-allocated
tensors; one replicating the entire per-frame pipeline including all the OpenCV
pre/post-processing at both 1080p and 4K; one splitting session creation and
`Run()` across threads the way OBS does. **All three stayed flat.**

The reason: the prebuilt library contains exactly one KleidiAI symbol,
`sqnbitgemm_neon::UseKleidiAI` — 4-bit quantized GEMM. `ArmKleidiAI::MlasConv` is
not in it. Same version number, different build configuration.

`buildspec.props`, which pins ONNX Runtime to build from source at `v1.23.2`, was
added on 2026-05-27. It is absent at the 1.4.0 tag — but 1.4.0 leaks identically,
so the KleidiAI-enabled ORT predates that file.

### What is NOT the cause

Ruled out by measurement, not assumption:

- **CoreML execution provider** — device was `cpu` throughout; `owned unmapped (neural)`
  is 0 K.
- **Capture path / GPU** — `IOSurface` 188 MB and `IOAccelerator` 185 MB, both flat,
  while 36.4 GB of a 36.7 GB footprint sits in `DefaultMallocZone`.
- **Filter settings / the update path** — reproduces with the properties dialog closed
  and no parameter ever changed.
- **The plugin's own buffer management** — `getRGBAFromStageSurface` balances
  map/unmap, textures are destroyed on every exit path, the `cv::Mat`s are refcounted
  and reassigned. Confirmed by the reproducers above, which exercise all of it.
- **ORT arena growth** — the reproducers show 33.6 MB of warm-up and then a plateau.
  That is what a bounded arena looks like. 18 GB/min with no plateau is not that.

---

## Result

| | stock 1.4.1 | rebuilt against prebuilt ORT |
|---|---:|---:|
| RSS at ~60 s | 17.74 GB | **0.76 GB** |
| Growth | ~18 GB/min | 0.02 GB over 61 s |
| Plugin heap total | 11.12 GB / 89,954 blocks | 22.7 MB / 9 blocks |
| `ArmKleidiAI::MlasConv` frames | 7,798 calls / 10.96 GB | 0 |

Inference still runs on the **CPU** provider with the same RVM model and identical
tensor shapes, so mask quality is unchanged. Convolution stacks are still present in
`malloc_history`; only the KleidiAI frames are gone.

---

## Reproducing the build (macOS arm64)

Needs `cmake`, `ninja`, Xcode command line tools, and OBS.app installed. It does
**not** need OBS or ONNX Runtime built from source.

```bash
WORK=~/obs-bgr-build && mkdir -p "$WORK" && cd "$WORK"

# 1. Prebuilt ONNX Runtime (this is the whole fix)
curl -sSL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz | tar xz
mv onnxruntime-osx-arm64-1.23.2 ort
ln -sfn . ort/include/onnxruntime          # its CMake config expects include/onnxruntime/

# 2. OpenCV 4 via vcpkg (usually a binary-cache hit, ~1 min)
git clone --depth 1 https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
./vcpkg/vcpkg install "opencv4[core,jpeg]" --triplet arm64-osx

# 3. OBS headers only -- no OBS build
curl -sSL https://github.com/obsproject/obs-studio/archive/refs/tags/31.1.1.tar.gz | tar xz

# 4. Shims so find_package(libobs) resolves to OBS.app's frameworks
mkdir -p shim/include shim/libobs shim/obs-frontend-api
cat > shim/include/obsconfig.h <<'EOF'
#pragma once
#define OBS_DATA_PATH "../../data"
#define OBS_PLUGIN_DESTINATION "obs-plugins"
#define OBS_RELEASE_CANDIDATE 0
#define OBS_BETA 0
EOF
FW=/Applications/OBS.app/Contents/Frameworks
cat > shim/libobs/libobsConfig.cmake <<EOF
add_library(OBS::libobs SHARED IMPORTED)
set_target_properties(OBS::libobs PROPERTIES
  IMPORTED_LOCATION "$FW/libobs.framework/Versions/A/libobs"
  IMPORTED_LOCATION_RELWITHDEBINFO "$FW/libobs.framework/Versions/A/libobs"
  INTERFACE_INCLUDE_DIRECTORIES "$WORK/obs-studio-31.1.1/libobs;$WORK/shim/include")
set(libobs_FOUND TRUE)
EOF
cat > shim/obs-frontend-api/obs-frontend-apiConfig.cmake <<EOF
add_library(OBS::obs-frontend-api SHARED IMPORTED)
set_target_properties(OBS::obs-frontend-api PROPERTIES
  IMPORTED_LOCATION "$FW/obs-frontend-api.dylib"
  IMPORTED_LOCATION_RELWITHDEBINFO "$FW/obs-frontend-api.dylib"
  INTERFACE_INCLUDE_DIRECTORIES "$WORK/obs-studio-31.1.1/frontend/api")
set(obs-frontend-api_FOUND TRUE)
EOF

# 5. Build the plugin
git clone https://github.com/royshil/obs-backgroundremoval.git src && git -C src checkout 1.4.1
cmake -S src -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_PREFIX_PATH="$WORK/shim/libobs;$WORK/shim/obs-frontend-api;$WORK/ort;$WORK/vcpkg/installed/arm64-osx"
cmake --build build

# 6. Make the bundle self-contained, then install
P=build/obs-backgroundremoval.plugin
mkdir -p $P/Contents/Frameworks
cp ort/lib/libonnxruntime.1.23.2.dylib $P/Contents/Frameworks/
install_name_tool -add_rpath "@loader_path/../Frameworks" $P/Contents/MacOS/obs-backgroundremoval

# Quit OBS first.
DST="$HOME/Library/Application Support/obs-studio/plugins/obs-backgroundremoval.plugin"
rm -rf "$DST" && cp -R $P "$DST"
```

Verify:

```bash
nm -arch arm64 "$DST/Contents/Frameworks/libonnxruntime.1.23.2.dylib" | grep -c ArmKleidiAI   # want 0
while :; do ps -o rss= -p $(pgrep -x OBS); sleep 5; done                                       # want flat
```

### Caveats

- The bundle is **unsigned**. Gatekeeper may object depending on how OBS is launched.
- It will be **overwritten by any plugin update**. If the growth returns after an
  update, that is why — rebuild to restore.
- Deployment-target warnings during linking (vcpkg objects built for macOS 26.0,
  linked at 12.0) are expected and harmless. Raise
  `-DCMAKE_OSX_DEPLOYMENT_TARGET` to silence them.
- Only tested on macOS 26.5.2, M5 Pro, OBS 32.2.1, plugin 1.4.1, RVM model, CPU device.

## How this was diagnosed

`vmmap -summary` for region attribution, `heap -sortBySize` for per-binary
attribution and block-size histograms, `leaks` to establish reachability,
`MallocStackLogging` + `malloc_history -allBySize` for allocation stacks, and the
release dSYM with `atos` to symbolicate a stripped binary. Standalone reproducers
were used to isolate ONNX Runtime and OpenCV from the OBS environment.

This investigation was carried out with the assistance of an AI coding agent.
Every number above is measured output, and several intermediate hypotheses it
formed (an ORT memory-pattern bug, a session-recreation leak, a cross-thread
issue) were wrong and are recorded as refuted rather than quietly dropped.
