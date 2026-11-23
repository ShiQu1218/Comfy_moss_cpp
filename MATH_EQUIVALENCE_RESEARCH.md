本次研究結論：**首次 LoKr 合併優先研究快速 ConvRot；完整 SLG 採樣優先研究相同分支的共同前段。** 兩者解決不同階段，不能相加成整體加速倍率。只有 Simple SLG、且 LoKr 快取已命中的既有工作流程，幾乎不會受益於這兩項。

日期：2026-11-16。依據目前工作區及 MossInfer commit `f4f38f5e90d9990795d125b357574a68aadbe924`。本次新增研究文件與獨立數值檢查腳本，未改動正式推論、helper、模型或快取。

「數學等價」在本文指實數代數上的同一運算，不代表 FP32、BF16、INT8 的完整執行結果相同。浮點加法不具結合律；改變歸約順序或 FMA 都可能改變結果。[NVIDIA 浮點運算說明](https://docs.nvidia.com/cuda/floating-point/index.html)

特別是目前的量化流程：即使旋轉矩陣恆等式成立，也不能據此證明兩段包含浮點捨入與量化的程式等價。如果需求仍是逐位元一致，以下重排算術的方案不能直接取代現有路徑，應沿用 [既有逐位元研究](HARDWARE_BITWISE_RESEARCH.md) 的驗收標準。

| 優先順序 | 變換 | 真正受益範圍 | 等價條件與限制 |
| --- | --- | --- | --- |
| 1 | 密集 ConvRot → regular-H4 butterfly | LoKr／完整 diff 首次建立快取 | 同一旋轉矩陣；改變累加順序 |
| 2 | 在旋轉座標直接合併，符合形狀時只旋轉 LoKr 因子 | 普通線性 LoKr 首次合併 | 分配律與正交性；量化結果須重新驗證 |
| 3 | 完整 SLG 共用首次跳層前的正向分支 | 啟用完整 SLG 的每次推論 | 相同輸入、conditioning、權重、位置資訊與執行語意 |
| 4 | 在 velocity 空間合併 CFG／無 rescale 的 SLG | 每次 Python guidance | 只減少張量運算，不省 DiT；效益預期較小 |
| 後續 | DoRA 逐層／分塊直接合併 | DoRA 首次建檔的 RAM／磁碟 I/O | 須保留原生正規化軸、epsilon 與 patch 順序 |

**1. 最小且明確的演算法機會：LoKr helper 的快速 ConvRot**

`native/lokr_merge.cu` 的 `rotate` 對每個輸出元素逐項累加 g 個輸入，前後各做一次。N×K 權重的旋轉成本為 O(NKg)。目前支援 g=4、16、64、256、1024。

固定版本上游 `src/weights/quant/quant_codec.cpp::hadamard_sign` 定義的矩陣是：

```text
H4 = [ 1  1  1 -1
       1  1 -1  1
       1 -1  1  1
      -1  1  1  1 ]
Hg = (H4 / 2) ⊗ ... ⊗ (H4 / 2), 共 log4(g) 個因子
Hgᵀ = Hg，Hg² = I
```

按 base-4 索引逐軸做 H4 butterfly，可將成本降為 O(NK log4 g)，不需 g×g 的旋轉表。g=256 時，是四個 butterfly 階段取代每個輸出的 256 項累加。這是運算結構的差異，不是實測加速倍率；記憶體存取、同步、配置、因子展開與磁碟時間仍然存在。

可重用上游 `src/kernels/quant/rotate_quantize.cu::h4` 與 `hadamard256` 的數學及資料布局。該檔已在**推論 activation** 使用快速變換，因此不能再把 FWHT 當成每層推論的全新收益。helper 使用 FP32 權重、不同分組與量化流程，不能直接呼叫整個 BF16 activation kernel。

也不能直接套常見 Sylvester-H2 FWHT：此處矩陣的符號及排列不同，須證明對應的排列／符號轉換，或直接沿用 regular-H4。外部 [Dao-AILab 實作](https://github.com/Dao-AILab/fast-hadamard-transform) 可作 GPU 實作參考，本專案已有可用結構，無需先新增套件。

實作範圍是 `native/lokr_merge.cu`、helper 建置與 `tests/check_gpu_merge.py` 的候選數值驗證。既有逐位元測試不可偷偷放寬；研究版應和原 helper 分開比較。正式切換會改變 helper 雜湊，依 `lokr_cache.py` 現有規則使快取重建。

**2. 更大的代數簡化：省掉基底的來回旋轉**

令 R 為沿 K 維度的分組正交矩陣、Z=diag(scale)Q 為解碼後仍在旋轉座標的基底、Δ 為線性 adapter 差值。忽略浮點捨入時，現有重新量化前的運算為：

```text
(Z R + Δ) R = Z + Δ R                 因為 R² = I
```

可直接對 Δ 做一次旋轉，再加回 Z；不用先把整份基底反旋轉，再旋轉回去。多個線性 patch 可由分配律處理，但重排加總會引入另一種捨入差異，不能混入非線性 DoRA 再視為普通相加。

LoKr 又有 Δ=α(A⊗B)。設 A 為 a×b，B 為 c×d，K=bd；當 **d 是 g 的倍數**，R=I_b⊗R_d，因此：

```text
(A ⊗ B)(I_b ⊗ R_d) = A ⊗ (B R_d)
```

只需旋轉較小的 B，再由現有 `kron_add` 類型的 kernel 展開及加總。旋轉元素數從 abcd 降到 cd，旋轉部分減少 ab 倍；最終 N×K 權重的寫入、逐列最大值及重新量化仍不能省略。若 B 本身分解，也可再研究結合律的乘法順序，但須將因子展開的 FP64→FP32 捨入列入比較。

此簡化有明確形狀限制：現有 `tests/check_gpu_merge.py` 使用 B 的寬度 96，對 g=256 不符合上述條件，不能直接套公式。可先只做一般 butterfly，之後針對實際 adapter 形狀加入少量特化；不要先建一套通用張量收縮規劃器。

**不能直接採用的延伸：** `(A⊗B)x` 可改成 `A X Bᵀ`（x 以 row-major reshape 為 b×d），免建完整 Kronecker 矩陣；但將現有「合併後 INT8 重新量化」改成「基底加 LoKr side branch」，通常不再是同一個量化模型：`Quant(W+Δ)` 與 `Quant(W)+Δ` 不相等。普通 LoRA 的 side branch 已存在，也不代表 LoKr 能無條件比照。

**3. 對採樣本身較有意義：完整 SLG 共用相同前段**

`component.py::begin` 的完整 SLG rows 與正向 row 共用相同 conditioning；上游 `src/pipeline/session.cpp` 已按 conditioning 身分共用文字精煉結果，`MossDit::forward` 也已共用相同 latent 的 video embedding。但後續 42 個 blocks 仍依 rows 執行。

對相同 x、sigma、conditioning、位置資訊與權重，如果額外 SLG 分支第一次在第 j 層跳層，兩條分支在第 0 至 j−1 層完全相同。可只算一份完整 activation（包含影像及文字 tokens），在第 j 層前複製，之後各自繼續。這是同一次推論內消除重複運算，不是跨 timestep 預測。

以無 Simple SLG、CFG 雙分支、額外一條只跳第 22 層的完整 SLG 為例，假設各 row 每層成本相同：

```text
原本：42 正向 + 42 負向 + 41 跳層正向 = 125 row-blocks
共用：125 - 22 = 103 row-blocks
理想 block 計算量少 17.6%；只看這部分的速度上限約 125/103 = 1.21×
```

這不是端到端承諾。正負提示詞長度、批次效率、複製與 graph 捕捉可能抵銷收益。分支分歧後不能再共用 suffix，即使後面的權重相同，輸入 activation 已不同。**Simple SLG 的正負 conditioning 不同，不能如此共用；既有 Simple-only benchmark 不會得到這個 17.6%。**

實作需觸及上游 `moss_dit.cpp` 的 row 分組、`block_runs`、fork activation 及 graph key；不能只修改 Python prediction rows。批次形狀改變也可能改變 GEMM、normalization、q8 attention 的數值路徑。代數上的共用並不自動保證目前量化 kernel 的逐位元一致，須比較 fork 前後狀態及每條 velocity。

可先以完全相同 conditioning／latent／skip set 的重複 rows 做最小去重實驗，再評估前段共用。相同提示文字不等於相同 conditioning；跨步、不同 latent、不同 adapter 或不同 skip 前綴都不能復用。

**4. CFG／SLG 的 velocity 合併**

目前 `component.py::denoise` 先對每個 row 計算 `p_i=x−σv_i`，再由 `guidance.py::combine_predictions` 合併。無 rescale 時：

```text
p = p_n + c(p_c-p_n) + Σ s_i(p_c-p_i)
  = x - σ [v_n + c(v_c-v_n) + Σ s_i(v_c-v_i)]
```

可只在最後做一次 velocity→prediction 轉換，減少張量配置與逐元素運算。這不省任何 DiT 評估、共享記憶體傳輸或模型載入，排序低於前述方案。`cfg=1` 已省負向 DiT 分支，不算新機會。

`rescaling_scale != 0` 時，原程式逐個 callback 以 `std(p_c)/std(p)` 縮放當前結果；不能用 velocity 的標準差代替，因為 x 在各位置不是常數。多個 rescale 也不能合併成單一線性係數。若實作，影響 `component.py`、`guidance.py`、`tests/test_guidance.py` 與端到端逐步比對，必須保留順序。

**5. DoRA：先消除完整 diff 的落盤，不先改正規化公式**

`lora.py::full_delta` 目前算出 patched−base，再輸出 FP32 diff，helper 讀回後加到 base 並重新量化。直接逐層把 patched 交給合併程序可減少大型暫存與一次相減／相加，但會改變 IPC 及錯誤恢復，也可能改變數值結果，故列為後續。

不能假設 DoRA 永遠沿輸出列正規化。本機 ComfyUI `comfy/weight_adapter/base.py::weight_decompose` 依 dora_scale 形狀分兩條路：輸出軸分支使用原 weight 的 norm；另一分支使用 weight_calc 沿另一軸的 norm；兩者加 dtype epsilon，再處理 strength。必須遵守這個實作，不能直接代入一般論文公式。若 norm 跨越切塊軸，需要完整統計或兩次掃描；逐塊各算 norm 會改變結果。多個 DoRA 與 LoKr 仍須依原 patch 順序執行。

**本次數值檢查**

新增 [scripts/check_math_equivalence.py](scripts/check_math_equivalence.py)，使用既有 NumPy 2.3.2、固定 seed 71、CPU 合成資料，不載入模型。實際通過：五種 g 的正交性、H4 butterfly 與密集矩陣對照、旋轉消去、對齊的 Kronecker 因子旋轉、Kronecker matvec、CFG／SLG 代數式，以及量化半整數邊界反例。FP64 隨機對照使用 rtol=atol=1e-12，另有 dyadic／整數資料的精確相等檢查；它們驗證實作樣本，不取代上述代數推導。

以 32 列、每列 2g 個元素，比較「原順序兩次密集旋轉」及「兩次 butterfly」，兩者都含同一份合成完整 diff：

| g | 重新量化前最大絕對差 | 不同 INT8 元素／總數 | 不同 scale 位元／32 列 |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0 / 256 | 0 |
| 16 | 3.58e-7 | 0 / 1,024 | 20 |
| 64 | 5.66e-7 | 0 / 4,096 | 22 |
| 256 | 1.43e-6 | 0 / 16,384 | 28 |
| 1024 | 2.86e-6 | 2 / 65,536 | 31 |

旋轉座標直接合併在 g=256 的最大差同為 1.43e-6，27/32 個 scale 不同；g=1024 有 2 個 INT8 值及 32/32 個 scale 不同。**即使 INT8 bytes 相同，只要 scale 不同，解碼權重也已不同。** 另外只需把固定 row maximum 下的 0.5 增加一個 FP32 ULP，就能把量化結果從 0 變 1。因此小誤差不保證量化或圖片不變。

重現方式：

```powershell
& 'D:/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe' scripts/check_math_equivalence.py
```

這些數字只描述小型 CPU 合成實驗，不是 GPU helper、真實 LoKr、完整模型或視覺品質的量測。腳本不提供速度比較，避免把 Python／NumPy 執行成本當成 CUDA 候選效能。

**下一輪最小實驗與效益界線**

先只實作獨立研究版 g=256 regular-H4 helper kernel，保留相同 adapter 加總、dtype、量化、CPU fallback 與錯誤檢查；比較原 rotate、butterfly、旋轉座標合併三組。分開量測 kernel CUDA events、含傳輸的 module 時間、含磁碟的完整建檔時間，並檢查峰值 RAM／VRAM。通過後才處理其他 g 或因子特化。

沿用 `tests/check_gpu_merge.py` 的正負 LoKr、多 adapter 與 diff 案例；另外驗證尾塊、零值、極小值、極值、量化邊界、QKV slice、非有限值拒絕，以及 GPU 失敗後不重複套用 patch。輸出比較包含 INT8、scale、反量化權重、每次 velocity、最終 latent 與 VAE 浮點結果；不能只看 PNG。若只接受逐位元相同，任一差異就判定不符合。

現有 `test_results/bitwise/merge.json` 記錄 helper 中位數約 10.65 秒，包含磁碟與其他準備，尚無旋轉佔比。設旋轉佔總耗時 p，旋轉自身加速 s，整體倍率最多為 `1 / ((1-p)+p/s)`；p 未量測前不能承諾省幾秒。LoKr 快取命中時，這項合併成本已被省掉。

FlashAttention 的分塊精確 attention 是另一種數學等價變換，但上游 exact 路徑已使用；high 的 q8 attention 則是另一個數值基準，切回 exact 不能當成等價加速。[FlashAttention 原論文](https://arxiv.org/abs/2205.14135) 因此不把「加入 FlashAttention」、已存在的 QKV／norm／RoPE 融合或文字精煉共用列為新收益。

未採用減少採樣步數、forecast_denoise、TeaCache、降低精度、線性 attention、直接合併不同 conditioning 或跨越非線性層等方案，因為無法保持指定計算的代數等價。本次未執行新 CUDA kernel、效能 profile、完整出圖或真實 DoRA 實驗，尚無新增端到端加速或品質不變的保證。
