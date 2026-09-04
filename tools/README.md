# 評価ツール（2026-08-30 追加）

スクラッチパッドではなくリポジトリ内に置く。セッションを跨いでも消えない。

```bash
cd tools
../.venv/bin/python scenegen.py     # -> scenes.json      (7シーン A〜G)
../.venv/bin/python poolgen.py      # -> scenes_pool.json (32シーンプール)

# A/B: 定数の上書きと名前付きパッチを組み合わせて全シーンを並列実行
../.venv/bin/python ab.py --scenes scenes_pool.json --tag base --jobs 14
../.venv/bin/python ab.py --scenes scenes_pool.json --tag c60 --jobs 14 \
    --patch com60 --set p:W_FLAT=600
```

- `harness.py` — agent を **同一プロセス内で**直接ドライブする（multiprocessing の
  タイムアウト機構を通さないので速い）。`fill_score` に加えて
  **`fill_loose`（inclusion_margin=+0.02 での充填率）**、積載個数、重心高さ、
  優先/ソフト貨物の埋没数、終了原因を返す。
- `ab.py` — `--set g:NAME=v` / `--set p:NAME=v` で geometry / packer の定数を
  上書きし、`--patch` で `patches.py` の名前付きパッチを当てて全シーンを並列実行。
  結果は `ab_<tag>.json`。
- `patches.py` — `_footprint_supported` の差し替えなど、構造的な変更の実験用。
- `diag_stuck.py` — 詰まった局面で止めて、レイアウト・空き体積・
  **実バリデータによる全探索**を出力する。

## 重要: 終了原因を必ず見ること

`ab_*.json` の `status` を集計すると、どの死に方をしたかが分かる。

| 終了原因 | 意味 |
| --- | --- |
| `transport` (`is_valid=False`) | 探索が候補を出せず、無検証のフォールバックを返して死んだ |
| `topple` (`is_placed_safe=False`) | 置いたが整定で崩れた |
| `DONE` | 全荷物を積み切った |

## 追加ツール（2026-08-30、GAMEPLAN.md 用）

| ファイル | 用途 |
| --- | --- |
| `survival.py` | 生存指標（生存型数 / オプション価値）と、それで上位K候補を再ランクする方策。**実測では割に合わなかった**（`docs/GAMEPLAN.md` §1.4）が、オフライン探索の評価関数として再利用する土台 |
| `diag_survival.py` | 1手ごとに「まだ置ける荷物型の数」を出力し、死の何手前に崩落するかを見る |
| `order_sens.py` | オフラインの荷物順序がどれだけ効くかを、ランダム順序との比較で測る |
| `budget_sens.py` | `optimize` の予算（15/60/150秒）を振って、予算が律速かどうかを測る |

```bash
cd tools
../.venv/bin/python diag_survival.py P05 P17
../.venv/bin/python order_sens.py
../.venv/bin/python budget_sens.py
```

## 実測済みの版と結果（32シーンプール）

| tag | 内容 | mean norm% | 積載率 |
| --- | --- | --- | --- |
| `P_base` | 現行提出物 | 17.82 | 32.6% |
| `P_bigfirst0` | `W_ITEM_VOL=0` | 15.19 | 30.6% |
| `P_nosupport` | 支持判定を撤廃 | 21.30 | 41.1% |
| `P_com60` | COM基準の支持判定 | 24.11 | 39.7% |
| `P_com60_flat` | ＋`W_FLAT=600` **（推奨）** | **25.47** | **40.7%** |
| `P_grad3` | ＋オプション価値 re-rank | 25.31 | 39.7% |
| `P_surv3` / `P_surv6` | ＋二値生存 re-rank | 24.65 / 24.87 | 39.1 / 39.5% |

生データは `ab_<tag>.json`。

## Day 3 で追加したもの

| ファイル | 用途 |
| --- | --- |
| `offgen.py` | オフライン専用シーンプール（`scenes_off40.json`、40シーン）を生成 |
| `probe_roll.py` | rollout のコストが「最初の失敗の前後」でどう分かれるかを測る |
| `probe_lns.py` | `optimize` の探索軌跡（試行数・改善が起きた試行番号・プレフィックス長）を出す |

### エージェント2本を A/B する

`ab.py --agent-dir` は `AGENT_DIR` 環境変数で `harness.py` に伝わる。
**`harness.py` を直接 import する前に `AGENT_DIR` を設定すること** — 以前は
`harness.py` が無条件で `agents/submit` を `sys.path` 先頭に挿していたため、
`--agent-dir` が黙って無視され、A/B が同一エージェント同士の比較になっていた。

```bash
cd tools
# 攻め(LNS)と守り(fixed)を40シーンのオフラインプールで比較
sed "s/^OFFLINE_SEARCH = 'lns'/OFFLINE_SEARCH = 'fixed'/" \
    ../agents/submit/agent.py > /tmp/x/submit/agent.py   # ほか2ファイルもコピー
../.venv/bin/python ab.py --scenes scenes_off40.json --tag fixed --jobs 16 \
    --optimize-budget 150 --agent-dir /tmp/x/submit
../.venv/bin/python ab.py --scenes scenes_off40.json --tag lns --jobs 16 \
    --optimize-budget 150
```

**注意**: `optimize` は壁時計の締切を使うので、この比較は CPU を飽和させた状態で
並列実行すると試行数が変わる。`--jobs` はコア数以下にすること。

### 本番相当の実行

```bash
PYTHONPATH=$PWD/.. ../.venv/bin/python ../scripts/run_test.py \
    --config-path ../configs/sample_config.json --module-path agents/submit/ \
    --result-dir /tmp/res/ --result-fname prod.json
```

`--module-path` はモジュール名に変換される（`/` → `.`）ので、
**ディレクトリ名にドットを含めないこと**（`submit.bak.1234` は `ModuleNotFoundError` になる）。

## 詰まりの原因を切り分ける（Day 3 以降）

| ファイル | 出力 |
| --- | --- |
| `diag_fn.py` | 詰まった局面でエージェントが棄却した候補を実バリデータで検査し、**どちらのゲート（幾何オラクル／支持判定）が偽陰性を出しているか**を数える |
| `diag_cause.py` | 全候補を棄却理由で分類（内包NG / 押し込み(Y)衝突 / 横スライド(X)衝突 / **高さ超過** / 支持不足） |
| `diag_skyline.py` | 停止時のスカイライン高さマップ。空きが「面積」なのか「使える形」なのかが一目で分かる |

```bash
cd tools
../.venv/bin/python diag_fn.py      C_lookahead1 scenes.json
../.venv/bin/python diag_cause.py   A_1c_offline scenes.json
../.venv/bin/python diag_skyline.py A_1c_offline scenes.json
```

## シーン集合の役割分担

| ファイル | 用途 |
| --- | --- |
| `scenes.json` (7) | A〜G。bigfirst の採否に使用済みで**汚染**。最終確認のみ |
| `scenes_pool.json` (32) | dev。チューニングはここで |
| `scenes_test.json` (32) | **ホールドアウト**。採用直前の再現確認にだけ使う |
| `scenes_off40.json` (40) | オフライン専用。`optimize` の比較用 |
| `scenes_offline.json` (16) | 同上の小型版 |

**スイープの最大値をそのまま採用しないこと。** 別集合で再現を確認する
（`docs/RESEARCH.md` §5）。

## 先読みプールの実効価値を測る

`scenes_pool_k1.json` は `scenes_pool.json` のうち `look_ahead > 1` の16シーンについて、
**荷物列はそのままで先読みだけ 1 に落とした**もの。両方を回して差を取ると、
「プールに選択の余地があることが実際にいくら効いているか」が分かる。

```bash
cd tools
../.venv/bin/python ab.py --scenes scenes_pool.json    --tag full --jobs 14
../.venv/bin/python ab.py --scenes scenes_pool_k1.json --tag k1   --jobs 14
```

2026-08-30 の実測（`ab_S_left150.json` vs `ab_POOL_k1.json`）:
先読みあり 29.98 / 積載 40.7%、k=1 強制 27.17 / 38.0% → **差はわずか +2.81pt**、
しかも **16シーン中4シーンで k=1 の方が良い**（最大 −12.0pt）。
選択肢が増えて成績が下がるのは論理的におかしいので、
`choose_action` のプール選択則に欠陥がある（`docs/SURVEY.md` §1）。

## Day 4 の診断ツール（2026-09-01）

**教訓: 探索の質を論じる前に、探索が予算内で完了しているかを測ること。**
この4本で「詰まりの原因は積付ヒューリスティクスではなく `choose_action` の予算配分」
という結論に到達した。

| ファイル | 用途 | この日わかったこと |
| --- | --- | --- |
| `diag_void.py` | 終了時のコンテナを5cmボクセル化し、スカイライン下の占有率とスカイラインより上のヘッドルームを分けて出す | 空きの大半は**スカイラインより上**（1〜2m³）。下は85〜93%埋まっている。→「隙間を詰める」系の施策は的外れ |
| `diag_gap.py` | 詰まった局面で `choose_action` に**大きな予算**を与え、本番予算との差を見る。あわせて (x,y) 両方向のグリッド全探索も試す | P00 は 20秒なら4通りの設定すべてで配置が見つかる。本番の3.5秒では見つからない = **時間切れで自死していた** |
| `diag_prof.py` | 詰まった局面での `best_placement` 1回の内訳を cProfile で出す | 1回 0.25〜0.35秒。`_landing` が46%。→ `MIN_CALL_BUDGET` と `_landing` 高速化の両方がここから出た |
| `diag_equiv.py` | 最適化前後の `_landing` を大量のランダムクエリで突き合わせ、出力一致と速度比を出す | 90000クエリで**不一致0件・4.81倍** |

## 物理ツール（`twin_*.py`）

`agents/submit/twin.py`（評価基盤の第3ゲート＝整定判定の複製）の校正・検証用。
**ツインは正確だがスコアには効かなかった**（`docs/WORKLOG.md` Day 4）。再挑戦の前に読むこと。

| ファイル | 用途 |
| --- | --- |
| `twin_bench.py` | ツインの構築コストと settle ステップ数ごとの所要時間 |
| `twin_valid.py` | 実際に転倒で死んだエピソードを再生し、ツインがその手を検出できるか（**12/12**） |
| `twin_roc.py` | 全ステップでツイン判定と実判定を突き合わせ、閾値ごとの検出率と誤棄却率を出す |

校正で判明した必須条件（どれか欠けると精度が落ちる）:
**ソフト荷物の `contactStiffness`/`contactDamping`/`linearDamping`**（無いと 5/12 しか当たらない）、
重力 `-9.8`（`-9.81` ではない）、`deterministicOverlappingPairs=1`、
扉側（外向き -Y）の平面には壁を立てないこと（開口部なので荷物は落ちる）。
