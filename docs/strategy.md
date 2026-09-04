# NEDO Challenge「Baggage-Loading Robot」コンテスト2 積付アルゴリズム — 金メダル（懸賞金候補者）獲得のための戦略レポート

## TL;DR
- **勝ち筋は「DRL単独ではなく、EMS/Extreme-Point ベースの候補生成 → 高さマップによる搬入経路つき配置判定 → 焼きなまし/ビームサーチによる順序・配置最適化」というヒューリスティック最適化路線**である。8秒/手・4vCPU・16GB・Python・非公開最終評価というこの環境は、学習済みDRLよりも「軽量シミュレータ上での多数ロールアウト＋メタヒューリスティクス」が有利で、AHC（AtCoder Heuristic Contest）型の勝ち方がそのまま効く。
- **最重要の実装資産は「自作ローカルシミュレータ」**である。公式評価（コンテナ内包判定、1.5cmクリアランスの直線押し込み干渉チェック、8cm上からの落下定着、動的安定性）を高速に近似再現する高さマップ＋簡易物理を numpy/numba で作り込み、内部代理目的関数（充填率・低重心・ルール違反・安定性の重み付き和）をリーダーボード（LB）と回帰で合わせ込むことが順位を決める。
- **過学習（暫定→最終のシーンシャッフル）対策として、自前の多様なシーン生成による交差検証（CV）を主指標にし、LBは補助検証フォールドとして扱う**。最終提出2枠は「LB最良の攻め型」と「自作CVで最も分散が小さい安定型」を選ぶ。

## Key Findings

**1. 課題構造とスケジュール（公式資料ベース）**
経産省・国交省連携事業「NEDO Challenge, Baggage-Loading Robot ～空港の未開拓領域に挑め～」のコンテスト2「積付アルゴリズム」は、SIGNATEプラットフォーム上で **2026年7月7日（火）14:00〜2026年10月19日（月）23:55** にスクリーニング（公開競技）、**2026年11月** にスクリーニング結果通知、**2026年11月〜12月** に成果審査（非公開）、**2027年2月** に受賞者決定・表彰、という日程で行われる。上位10者程度が懸賞金候補者として成果審査に進み、成果審査はバーチャル審査・実地審査・書面審査による総合評価で、**実地審査は佐賀空港等の試験ライン**を想定する（参加者の出席義務なし）。公開競技は5か月程度、難易度の異なる複数課題が用意される。運営事務局は株式会社三菱総合研究所。実装言語はPythonが予定されている。成果審査時には「開発成果報告書」（開発方針、アルゴリズム構成、機能要件対応状況、拡張性等）の提出も求められる点に注意。

**2. コンテナ（LD3: AKE/AKN）の幾何**
外寸は AKE・AKN とも **1,534(W)×2,007(L)×1,626(H) mm**。ただし床面フットプリントは **1,534×1,562 mm** で、天井（ルーフ）長は **2,006〜2,007 mm**。PalNet の LD-3 技術データシート（IATA Designator AKE、設計仕様 NAS 3610-2K2C、P/N 8136 0000）は原文で「Base Size 60.4" x 61.5" / 1,534 mm x 1,562 mm；Height 64" / 1,626 mm；Roof Length 79" / 2,006 mm；Tare Weight from 172 lbs / 78 kg；Max. Gross Weight 3,500 lbs / 1,588 kg」と明記する（最大総重量 1,588 kg は DoKaSch の AKE(LD3) 仕様ページでも確認）。床と天井長の差 **≈444 mm（2,006−1,562）** が機体胴体に沿う斜めカット（ノッチ、IATAコンター"E"）の水平方向の広がりに相当する。内容積は **AKE ≈4.3 m³、AKN ≈4.1 m³**。内寸の一例は 1,920×1,450×1,620 mm（Cargo-Planner）。**AKN はフォークリフトポケットを持つフォークリフト対応型**で、その分だけ有効容積がわずかに小さい。扉開口は概ね 1,410〜1,427(W)×1,528〜1,533(H) mm でノッチと反対側の垂直面にあり、幌（フルファブリック）扉と金属扉が存在する。**斜めカットの角度・垂直方向高さは商用スペックシートには明記されておらず**（IATA ULD Technical Manual のコンター"E"図面が唯一の厳密ソースだが有料・非公開）、コンペでは公式がCSVで「コンテナ内側の利用空間の立体形状データ（扉からの距離や上方に確保すべき空間距離等）」を与えるため、**与えられた有効空間データをそのままボクセル化／高さマップ化して扱うのが正解**。公式も「内寸は形式・個体で異なるので差異に対応せよ」と明記している。

**3. 手荷物分類とルール（公式）**
手荷物は直方体近似モデルで、外寸・重量・分類（ハード／変形ハード／ソフト（変形なし）／ソフト（変形あり））・優先手荷物フラグを持つ。公式要件は明確に、① 荷崩れしにくい低重心・隙間最小・はみ出し/傾き回避、② 充填率（人手で約9割が実現可能とされ、これに近い高さを、10%単位で指定可能に）、③ 重い/硬い荷物をソフト手荷物の上に置かない、④ 優先手荷物（ファースト・乗継）は取り出しやすいよう一般手荷物より上側、⑤ ロボット積込みのため配置位置の前方（手前）に**コンテナ外まで直線的に通過できる空間＋上方・側方の余裕（例：上10cm・左5cm・右5cm、個別設定可能）**を確保、⑥ 最大6台（コンテスト2では2台）の一体積付、⑦ 到着順（オンライン）積付対応、⑧ 並替対象数はEBSでコンテナ数台分（1台40個程度）、メイクエリアで数個〜1台分、と幅広い。優先手荷物を上へ・重い荷物を下へ・重心はコンテナ中央寄りに、という要求は、FAA AC 120-85B や IATA/航空会社の実務ガイド（「heavier and larger pieces on the bottom, lighter and fragile on top, CG in the ULD center zone」）とも整合する。

**4. 評価の物理検証（ユーザー提供の公式評価仕様）**
1手ごとにリアルタイム検証し、1つでもNGなら失敗：(a) アクション形式（座標・コンテナindexが許容範囲）、(b) コンテナ内包の幾何判定、(c) 手前から目標座標への直線押し込み軌道シミュレーションで既存荷物・壁・棚と衝突（1.5cm以内接近）で失敗、(d) 目標の8cm上から落として物理を少し進め、初期位置からのずれで定着確認。完了後の総合評価（100点満点の重み付き平均）は、空間充填率・重心の低さ・積付ルール/破損リスク（優先・ソフトの上に一般荷物でペナルティ、優先専用コンテナ違反もペナルティ、ソフトと優先は独立評価）・動的安定性（蓋をして重力を動的変動、揺れ収束後のずれ・発生した力/運動エネルギー計測）。暫定評価は評価シーンの一部、最終評価は残り。投稿1日5回、最終提出2ファイル。

**5. アルゴリズム研究の到達点**
- オフライン3D-BPP/CLP: Extreme Point（Crainic et al. 2008）、Empty Maximal Space/EMS（Parreño et al. 2008 のGRASP、Lai&Chan の difference process）、Deepest-Bottom-Left-Fill、Heightmap-Minimization（Wang&Hauser, ICRA 2019: 「a new 3D positioning heuristic called Heightmap-Minimization heuristic is proposed, and heightmaps are used to speed up the search」であり、占有体積増加を最小化する配置を選び、既存ヒューリスティクスを上回りロボット積付で最多の実行可能解を発見）、Wall/Layer/Block-building（George&Robinson 1980起源）、BRKGA（Gonçalves&Resende 2012, 2013、maximal-space表現で箱の順序と層タイプを進化）、ビームサーチ（Araya&Riff 2014、Lim の multi-round partial beam search）。
- オンライン3D-BPP: Zhao et al. AAAI 2021（制約付きDRL、feasibility maskのpredict-and-project、lookahead/multi-bin/re-orient拡張可）、PCT（Zhao et al. ICLR 2022、EMS/EP/コーナー点でツリー展開しTransformerで評価、連続空間で強い）、Deliberate Planning（PCT + テスト時物理シム＋プランニング、IJRR 2025）、GOPT（Transformer、2024）、Fast Stability Validation + Stable Rearrangement（arXiv 2507.09123、支持多角形とMCTS）。
- **実証的知見**: 論文はDRLがヒューリスティクスを平均性能で上回ると主張するが、Zhao et al.（AAAI 2021, arXiv:2006.14978）自身が原文で「imposing such constraints explicitly leads to inferior performance. We found that the performance (average reward) drops about 20% when adding such constraints」と報告する通り、DRLは制約の入れ方に敏感で、明示制約を課すと平均報酬が約20%低下する。実務コンペ（限られたCPU・Python・非公開評価）では、DRLの学習コスト・環境再現の難しさ・過学習リスクが大きく、巧妙なヒューリスティクス＋探索が費用対効果で勝りやすい。

**6. 物理的安定性の近似判定**
静的安定性は「重心が支持接触点の凸包（支持多角形）内にあるか」（Ramos et al. 2016、static mechanical equilibrium）が必要十分に近い基準で、簡易版の支持面積率（底面接触率、Gehring&Bortfeldt 1997、閾値66%等）より高精度。One4Many-StablePacker 等は support ratio≥0.66 と「重い箱を軽い箱の上に載せない（重量比3倍ルール）」を制約化。動的安定性はPyBullet等で settling をシミュレートしずれ量・速度閾値で崩壊検知（friction≈0.75、重力9.81 が実装例）。テスト時物理シムは学習より検証に必須という知見（Deliberate Planning: quasi-static推定で訓練し、テスト時シムで摂動強度 k_d を上げるほど安定率が向上し k_d=8 で100%）。

## Details

### A. 全体アーキテクチャ：3層構成のハイブリッド・エージェント

課題A/B/Cを1つのコードベースで捌くため、以下の3モジュールに分離する。

```
[1] 候補生成器 CandidateGenerator(state, item)
      -> List[(container, x, y, z, orientation)]   # 実行可能な離散候補（EMS/EP + 高さマップ）
[2] 評価器 Evaluator(state, item, placement)
      -> surrogate_score (float)                    # 代理目的関数（内包/干渉/安定/ルール/充填/重心）
[3] 探索/順序最適化 Planner
      - 課題A: オフラインで順序を焼きなまし/ビームサーチ最適化
      - 課題B: プール(3-40)からの選択をビームサーチ + ロールアウト
      - 課題C: 純オンライン、候補×評価のgreedy + 1-2手lookaheadロールアウト
```

**状態表現（コア）**：各コンテナを高さマップ `H[nx][ny]`（xy平面グリッド、セルサイズ 2〜5cm、値は各(x,y)列の占有上端z）で保持。加えて、幾何判定精度のため各配置済み荷物の直方体リスト（AABB）と、支持関係グラフ（誰が誰を支えているか）、質量分布を持つ。斜めカット部・棚・上方確保空間は「配置禁止/高さ制約マスク」`Cmask` あるいは各(x,y)での許容zレンジとして前計算する。ソフト（変形あり）は高さマップ上で圧縮率を持たせる（例：上載重量に応じ厚みを最大α%減）。

**行動空間**：`(container_id, x, y, orientation)`。zは高さマップから一意に決まる（後述の最下降z）。回転は直方体なので基本は水平面のyaw 2通り（長辺/短辺）＋必要に応じ6面姿勢だが、**優先手荷物の取り出しやすさ・安定性から寝かせ配置中心に絞る**とオンライン8秒制約下で候補が現実的になる。

### B. 高さマップによる配置判定（最下降z・内包・支持）

Wang&Hauser / SDF-Pack と同じく、コンテナ上面視の高さマップ `H_c` と荷物の底面高さマップ（直方体なら平坦）から、衝突しない最深zを

```
z(x,y,r) = max over (i,j) in footprint(r) of H_c[x+i, y+j]
```

で O(セル数) で求める。直方体なので footprint 内の H_c の最大値そのもの。内包判定は `z + h(r) ≤ Z_local(x,y)`（斜めカット・上方確保を反映した局所上限）。**支持率**は底面グリッドのうち `H_c[x+i,y+j] == z`（=接触）セルの割合、**支持多角形判定**はその接触セル集合の凸包に荷物重心(x,y)投影が入るかで行う（numbaで凸包内包を高速化）。

### C. 搬入経路（直線押し込み）干渉チェック — 本コンペ最大の差別化点

公式は「手前側からコンテナ外まで直線的に押し込む軌道」で干渉（1.5cm以内接近）したら失敗、上10cm/左5cm/右5cmの余裕を要求する。これを**スイープ体積（swept volume）の高さマップ投影**で高速判定する：

- 押し込み方向を +y とする（扉→奥）。荷物を目標(x, z)に置いたまま y をコンテナ手前端から目標yまで動かす掃引体は、**xz断面を押し出したトンネル**になる。
- クリアランス込みで荷物を左右+5cm/上+10cm/近接1.5cm 膨張（Minkowski和的マージン）させた断面が、手前側の各yにおいて既存高さマップ／壁／棚と交差しないことを確認。
- 実装は「目標より手前(y'<y_target)の領域で、荷物のxスパンにわたる `H_c` の最大値が、荷物底面z（＋1.5cmマージン）より低い」ことのチェックに帰着。つまり **手前が目標より高くないこと**。numpyのスライス最大でO(セル数)。
- これにより「奥から手前へ階段状に積む（far-to-near / 高い所を奥に）」配置が自然に誘導される（Zhao et al. の far-to-near報酬と同型の効果）。**候補生成の段階でこの制約を満たす配置のみ残す**ことがタイムアウト回避と高得点の両立に効く。

擬似コード（1手の評価）：
```python
def feasible_and_score(state, item, cont, x, y, r):
    z = heightmap_drop_z(state.H[cont], item, x, y, r)      # 最深z
    if not inside_container(cont, x, y, z, item, r):        # 斜めカット/上方確保込み
        return None
    if not straight_insertion_ok(state.H[cont], item, x,y,z,r,
                                 clear=(0.015, 0.05, 0.05, 0.10)):  # 近接/左/右/上
        return None
    supp = support_ratio(state.H[cont], item, x, y, z, r)
    if supp < TAU_SUP: return None                          # 支持不足を早期棄却
    if not cog_in_support_polygon(...): return None
    return surrogate(state, item, cont, x, y, z, r, supp)
```

### D. 内部代理目的関数（surrogate objective）の設計

最終スコア（充填率・低重心・ルール・動的安定性の重み付き100点）を模す代理関数を、**1手ごとの増分**で設計する：

```
ΔS = w1·ΔFill − w2·ΔCoGHeight − w3·RulePen − w4·StabRisk − w5·WasteBuried
```

- **ΔFill**: 追加した荷物体積 / 有効容積（変形なし体積で評価される点に注意）。
- **ΔCoGHeight**: 全体重心zの上昇。重い荷物ほど低く置くと減少。増分計算 `z_cog = Σ m_i z_i / Σ m_i`。
- **RulePen**: 「ソフト／優先の上に一般（重い/硬い）」検出、優先が上側にない、優先専用コンテナ違反。ソフトと優先は独立に足す。
- **StabRisk**: (1−支持率) と、支持多角形余裕（重心投影から凸包境界までの最小距離の負値）、隣接接触面積の少なさ、はみ出し量。**動的安定性の代理**として「支持率が高く・重心が支持多角形の中央寄り・下段が上段より重い」ほど低リスク。
- **WasteBuried**: この配置により将来使えなくなる空隙（EMS体積の減少や、天井方向の分断）。SDF-Packの「占有体積増加最小化」や Wang&Hauser の Heightmap-Minimization の思想。

重み `w` は後述のチューニングで推定。**注意：これは代理であり、真の点数は物理シム結果**なので、最終判断は必ず自作物理シムのスコアで行う。

### E. 課題別の設計

**課題C（純オンライン、観測1、配置のみ）**
- 候補生成：現荷物に対し EMS/コーナー点＋高さマップで実行可能配置を全列挙（数百〜数千候補）。
- 評価：surrogate で採点。8秒あるので**1〜2手先の仮想ロールアウト**（次に来る荷物は未知なので、既知の分布からサンプルした代表荷物や「平均箱」を数個置いてみる greedy rollout）で上位K候補を再評価。
- 決定：ロールアウト平均が最良の配置。タイムアウト保険として、まず0.1秒で貪欲解を確保→残り時間で改善（anytime）。

**課題B（プール3〜40、選択の余地あり）**
- 「どの荷物を」「どこに」の二重決定。**ビームサーチ**（幅B=16〜64）で「プールから1個選び最良配置」を1ステップとし、数手先までの累積 surrogate で枝を保持。
- multiprocessing（4vCPU）で枝を並列ロールアウト。重い/大きい荷物を先に底へ、という事前ソートバイアス（BRKGAの random-key 的に順序をゆらす）。
- プール上限40なら、各手で候補（40×配置数）が大きいので、**荷物側は上位数個（重い・底面が広い・優先/ソフト制約が厳しい）に絞る**足切りが必須。

**課題A（オフライン順序最適化、40/80個、オンラインは順序固定で観測1）**
- オフライン3分：`順序（インデックス配列）` を決める。デコーダは「その順序で課題Cと同じgreedy配置器に流す」。
- 最適化は **BRKGA もしくは 焼きなまし**：解＝順序（＋各荷物の姿勢/優先コンテナ割当のrandom-key）、評価＝自作シミュレータでの完成スコア。
  - 焼きなまし近傍：2-opt/挿入/ブロック移動、姿勢反転、コンテナ間移動。温度は初期に受理率40〜50%、指数冷却で終盤ほぼ0。1評価が重い（フル配置）ので、**差分評価・部分再配置・評価器の高速化が生命線**。
  - ビームサーチを初期解生成、焼きなましで磨くハイブリッドが定石（AHC常套）。
- 3分でタイムアウトすると「元の順序」がそのまま使われる（＝最悪化）ので、**必ず時間監視して確定解を返す**。

### F. 高速化（4vCPU / 8秒・3分 / Python）

- **高さマップ演算を numpy スライス＋numba njit** に。最下降z・掃引干渉・支持率は全て2Dグリッドの max/比較で表現しベクトル化。numbaはNumPy配列＋プリミティブ型で C 近傍速度。
- **候補の早期枝刈り**：支持率・内包・掃引の順で軽い判定から。実行不可を最速で捨てる。
- **multiprocessing で並列ロールアウト**（4プロセス）。各プロセスに状態のコピー。GILを避けるためプロセス並列。
- **事前計算**：コンテナ形状マスク、姿勢別footprint、優先/ソフト判定テーブル。
- **anytime設計**：全モジュールに時間予算を持たせ、deadline 直前に必ず best を返す（タイムアウト時のランダム/元順序ペナルティを絶対に踏まない）。
- 物理シム（PyBullet）は**開発時のオフライン検証専用**にし、本番の8秒判断には使わない（重すぎる・再現性リスク）。本番は幾何＋静力学近似で代替。

### G. ローカルシミュレータ整備（最重要）

公式評価器を再現する自作環境を作る。構成：
1. **幾何エンジン**：内包・掃引干渉（1.5cmクリアランス、上10/左5/右5cm）・支持率。
2. **簡易定着（8cm落下）**：落下後、支持率不足なら重心が支持多角形へ向かってずれる量を近似（または落下先の最下降z補正）。閾値超のずれで失敗フラグ。
3. **PyBullet動的安定性**：蓋あり・重力動揺で settling させ、ずれ・運動エネルギーを測る。One4Many/Deliberate Planning流に、複数回・複数摂動強度 k_d で安定率を測り、代理 StabRisk と回帰して重みを合わせる。
4. **スコア集計**：充填率・重心・ルール・動的安定性を公式の重み付き平均（重み未知部分はLBから推定）で。

### H. チューニングと汎化（暫定→最終シャッフル対策）

- **自作シーンジェネレータで大量の多様なシーン**（棚有/無、初期空/既積、優先専用有/無、荷物構成の偏り、ソフト比率）を生成し、**交差検証（seed違い多数）を主指標**にする。LBはあくまで補助フォールド（Kaggleのpublic LBを追加foldとして扱う流儀）。
- 重み `w1..w5` と閾値 `TAU_SUP`、グリッド解像度、ビーム幅、温度スケジュールは**Optunaで自作CVスコアを最大化**。LB専用チューニングは禁物（shake-up＝順位大変動の温床）。
- **1日5投稿は情報採取に使う**：1変数ずつ変えてLBとローカルCVの相関を測り、代理関数の重み推定を較正。相関が崩れる=過学習の兆候。
- **最終2枠**：(i) LBとCV両方で最良の「攻め」型、(ii) CVでスコア分散が最小の「守り」型。最終評価が別シーンである以上、分散の小ささが金メダル残留率を決める。
- **崩壊系の絶対回避**：1手でもNG=失敗（0点系）になり得るので、探索中も「実行可能性を最優先、スコアは二番目」。実行不可を出さないことがまず前提。

## Recommendations

**フェーズ0（〜1週目）: 基盤**
- CSV入出力・座標系・姿勢・コンテナ形状（与えられた有効空間データのボクセル化）を確定。公式のaction形式に厳密準拠。
- 高さマップ状態表現と、内包・最下降z・掃引干渉・支持率の numpy/numba 実装。単体テストで幾何を検証。
- ベンチマーク閾値：1手の feasible判定が 1ms未満、候補全列挙が数十ms。

**フェーズ1（2〜3週目）: ベースライン**
- 課題C：EMS/EP候補＋greedy（surrogate）で「実行不可を絶対出さない」安定ベースライン。
- 課題A：与えられた順序をそのまま流すだけの安全提出（タイムアウト0）→ LB初期値取得。
- ベンチ：LBで「失敗ゼロ・充填率60%台」をまず確保。

**フェーズ2（4〜7週目）: 探索の導入**
- 課題B：ビームサーチ＋並列ロールアウト。課題A：BRKGA/焼きなましで順序最適化（anytime確定解つき）。
- PyBullet検証ループを構築、代理StabRiskを物理結果に回帰。
- ベンチ：充填率75〜85%、動的安定性ペナルティ最小化。人手の9割充填を目標線に。

**フェーズ3（8週目〜締切）: チューニングと汎化**
- Optunaで重み・解像度・ビーム幅・温度を自作CV最大化。多様シーンでの分散低減。
- LB×CV相関監視、shake-up耐性チェック。最終2枠（攻め型/守り型）を選定。

**判断を変える閾値**
- 自作CVとLBの相関 < 0.7 → 代理関数/重みが過学習。シーン多様化と重み再推定へ。
- greedyで実行不可発生率 > 0 → 探索より先に幾何/クリアランス実装を修正。
- 焼きなまし1評価が遅くオフライン3分で反復<数百 → numba化/差分評価/部分再配置を優先。
- DRLを試すのは、ヒューリスティクスで上位が頭打ちかつ学習環境が安定再現できた場合のみ（PCTコードを流用）。基本は非推奨。

## Caveats
- **物理シムの再現性リスク**：公式の物理エンジン（詳細非公開）と自作PyBulletは挙動が一致しない。自作CVを過信せず、LB較正とマージン（支持率・クリアランスを公式要求より厳しめ）で吸収する。8cm落下・動的重力揺動は特に差が出やすい。
- **LB過学習（shake-up）**：暫定と最終でシーンが違う設計は意図的に順位変動を起こす。公開LB最適化は罠。CV主義を貫く。
- **タイムアウト＝致命傷**：オフライン3分超で元順序、オンライン8秒超でランダム行動が強制され、いずれも大幅減点。全経路にdeadlineガードと確定解フォールバックを。
- **1.5cmクリアランス／掃引の実装漏れ**：内包はOKでも搬入経路NGで失敗、が頻発しやすい。掃引干渉を候補生成段階で満たすのが必須。
- **斜めカット・棚・優先専用コンテナ・既積初期状態**のバリエーション網羅漏れ。全バリエーションを自作シーンに含める。
- **ソフト手荷物の変形モデル**は公式の扱い（変形なし体積で充填率評価等）と自作近似がズレ得る。変形ありは上に載せない設計で回避を優先。
- **AKE/AKN内寸の個体差**：公式は「内寸は形式・個体で異なる、差異に対応せよ」と明記。ハードコードせず、与えられたコンテナCSVの形状データを常に使う。斜めカットの正確な角度・高さは商用ソースに無く、公式提供の立体形状データが唯一の確実な入力。
- **充填率9割・1.5cm/8cmクリアランス・日程等の一部数値は非公開のNEDO/SIGNATE公式資料に基づく**もので、第三者Webソースでの独立検証は不可。正式懸賞広告での確定値を必ず確認すること。

## 参考文献・リポジトリ・論文
- 公式: NEDO コンテスト2補足資料PDF https://www.nedo.go.jp/content/800052509.pdf ／ SIGNATE特設 https://service.signate.jp/campaign/baggage-loading-robot-2026 ／ 経産省 https://www.meti.go.jp/press/2026/07/20260707002.html ／ NEDO公募 https://www.nedo.go.jp/koubo/CD2_100451.html
- Zhao et al., Online 3D-BPP with Constrained DRL, AAAI 2021 https://ojs.aaai.org/index.php/AAAI/article/view/16155 ／ arXiv https://arxiv.org/pdf/2006.14978 ／ code https://github.com/alexfrom0815/Online-3D-BPP-DRL
- Zhao et al., PCT, ICLR 2022 https://openreview.net/forum?id=bfuGjlCwAq ／ code https://github.com/alexfrom0815/Online-3D-BPP-PCT
- Deliberate Planning of 3D-BPP on PCT, arXiv 2504.04421 https://arxiv.org/abs/2504.04421
- Online 3D-BPP with Fast Stability Validation and Stable Rearrangement, arXiv 2507.09123 https://arxiv.org/pdf/2507.09123
- GOPT (Transformer), arXiv 2409.05344 https://arxiv.org/pdf/2409.05344
- One4Many-StablePacker, arXiv 2510.10057 https://arxiv.org/html/2510.10057v1
- Parreño et al., Maximal-space GRASP for CLP, INFORMS JoC 2008 https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0254
- Crainic et al., Extreme Point heuristics, INFORMS JoC https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0250
- Gonçalves & Resende, multi-population BRKGA for CLP https://www.sciencedirect.com/science/article/abs/pii/S0305054811000827 ／ 2D/3D BPP BRKGA https://www.sciencedirect.com/science/article/abs/pii/S0925527313001837 ／ BRKGA実装例 https://github.com/dasvision0212/3D-Bin-Packing-Problem-with-BRKGA
- Wang & Hauser, Stable bin packing with a robot manipulator (Heightmap-Minimization), ICRA 2019, arXiv 1812.04093 https://ar5iv.labs.arxiv.org/html/1812.04093
- SDF-Pack, arXiv 2307.07356 https://arxiv.org/pdf/2307.07356
- Ramos et al., static mechanical equilibrium stability https://www.researchgate.net/publication/306416781
- Static stability vs packing efficiency, C&OR 2025 https://www.sciencedirect.com/science/article/abs/pii/S0305054825000334
- ULD仕様: PalNet LD-3データシート（NAS 3610-2K2C） ／ ACL Airshop https://www.aclairshop.com/container_specs.php ／ Unilode https://www.unilode.com/uld-specification/ ／ Cargo-Planner https://cargo-planner.com/guides/uld-types-and-dimensions/ ／ DoKaSch https://www.dokasch.com
- 航空貨物CoG/積付: FAA AC 120-85B https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-85B.pdf ／ IATA ULD解説 https://www.iata.org/en/publications/newsletters/iata-knowledge-hub/what-is-aircraft-uld-in-air-transport/
- AHC方法論: AtCoder Heuristic https://info.atcoder.jp/overview/contest/heuristic ／ 焼きなまし https://blog.oimo.io/tag/simulated-annealing/
- 高速化: Numba公式 https://numba.pydata.org/numba-doc/dev/user/vectorize.html