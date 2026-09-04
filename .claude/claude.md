# 手荷物自動積付コンペ 戦略ブリーフ（作業用）

> このドキュメントは実装着手のための要約版です。調査は公開文献ベースで行っており、**人間の専門家によるレビューは未実施**。末尾の「未検証の前提」を先に潰してから本実装に入ってください。

---

## 1. 何を作るか

LD3規格ULDコンテナ（AKE/AKN、最大2台）への手荷物積付エージェント。2つの機能を実装する。

| 機能 | 入力 | 出力 | 制限時間 |
|---|---|---|---|
| オフライン最適化 | 全手荷物リスト（40〜80個） | 積付順序（index配列） | 3分（超過→元の順序が採用） |
| オンライン最適化 | 観測プール（1〜40個）＋各コンテナ空き状況 | 荷物選択・対象コンテナ・3D座標・姿勢 | 1手8秒（超過→ランダム行動） |

出題は3パターン。

- **課題A**: オフラインあり → 順序固定でオンライン（観測1）
- **課題B**: オフラインなし、プール3〜40
- **課題C**: オフラインなし、観測1（配置のみ決定）

コンテナ条件のバリエーション: 台数1/2、棚有無、初期状態（空/既積）、優先手荷物専用コンテナの指定有無。

---

## 2. 評価の要点（ここを外すと即失点）

### 1手ごとの検証（1つでもNGなら積付終了＝そこで打ち切り）
1. アクション形式（座標・コンテナindexが範囲内）
2. コンテナ内包判定（斜めカット形状を含む）
3. **搬入経路の干渉チェック** — 手前から目標まで直線押し込み、既存荷物・壁・棚に **1.5cm以内接近で衝突扱い**
4. 定着確認 — 目標の **8cm上から落として** 物理を進め、ずれが大きいと失敗

### 完了後スコア（重み付き平均・100点）
- 空間充填率
- 重心の低さ
- ルール違反ペナルティ（ソフト／優先手荷物の上に一般手荷物、優先専用コンテナ違反。ソフトと優先は独立評価）
- 動的安定性（蓋をして重力を動的変動、収束後のずれ・力・運動エネルギー）

### 運用
- 暫定評価と最終評価で**シーンが異なる** → LB過学習は致命的
- 投稿1日5回、最終提出2ファイル
- 評価基盤: Ubuntu 24.04 / 16GB / 4 vCPU

---

## 3. 推奨アーキテクチャ

**DRLではなくヒューリスティクス＋探索を推奨。** 4vCPU・Python・8秒・非公開最終評価という条件では、学習済みモデルより「軽量シミュレータ上での多数ロールアウト＋メタヒューリスティクス」が費用対効果で勝る。

```
[1] CandidateGenerator(state, item)
      -> [(container, x, y, z, orientation), ...]   # EMS/Extreme-Point + 高さマップで実行可能配置を列挙
[2] Evaluator(state, item, placement)
      -> surrogate_score                            # 代理目的関数
[3] Planner
      課題A: 順序を焼きなまし / BRKGA で最適化
      課題B: ビームサーチ（幅16〜64）＋並列ロールアウト
      課題C: greedy ＋ 1〜2手先ロールアウト
```

### 状態表現
- コンテナごとに **高さマップ `H[nx][ny]`**（セル2〜5cm、値は各列の占有上端z）
- 併せて配置済み荷物のAABBリスト、支持関係、質量分布
- 斜めカット・棚・上方確保空間は「各(x,y)での許容zレンジ」として前計算
- ソフト（変形あり）は上載重量に応じた厚み圧縮を近似

### 行動空間
`(container_id, x, y, orientation)`。**zは高さマップから一意に決まる**（footprint内の`H`の最大値）。姿勢は水平yaw 2通りを基本とし、候補爆発を抑える。

---

## 4. 実装の核：3つの判定

```python
def feasible_and_score(state, item, cont, x, y, r):
    z = heightmap_drop_z(state.H[cont], item, x, y, r)   # footprint内のHの最大値
    if not inside_container(cont, x, y, z, item, r):     # 斜めカット/上方確保込み
        return None
    if not straight_insertion_ok(state.H[cont], item, x, y, z, r,
                                 clear=(0.015, 0.05, 0.05, 0.10)):  # 近接/左/右/上
        return None
    supp = support_ratio(state.H[cont], item, x, y, z, r)
    if supp < TAU_SUP:
        return None                                       # 支持不足を早期棄却
    if not cog_in_support_polygon(...):
        return None
    return surrogate(state, item, cont, x, y, z, r, supp)
```

### 搬入経路判定（最大の差別化点）
押し込み方向を +y（扉→奥）とすると、クリアランス込みで膨張させた荷物断面のトンネルが既存物と交差しないこと。実質的には

> **目標位置より手前(y' < y_target)の領域で、荷物のxスパンにおける `H` の最大値が、荷物底面z（＋1.5cm）より低い**

に帰着する。numpyのスライス最大でO(セル数)。この制約を**候補生成の段階で満たす**こと。結果として「奥から手前へ階段状に積む」配置が自然に誘導される。

### 支持判定
- 支持率 = 底面グリッドのうち `H[x+i,y+j] == z` のセル割合
- 支持多角形 = 接触セル集合の凸包。荷物重心のxy投影がその内部にあるか（numbaで高速化）

---

## 5. 代理目的関数

1手ごとの増分で設計する。

```
ΔS = w1·ΔFill − w2·ΔCoGHeight − w3·RulePen − w4·StabRisk − w5·WasteBuried
```

| 項 | 内容 |
|---|---|
| ΔFill | 追加体積 / 有効容積 |
| ΔCoGHeight | 全体重心zの上昇量（`z_cog = Σ m_i z_i / Σ m_i`） |
| RulePen | ソフト／優先の上に一般荷物、優先が上側にない、専用コンテナ違反 |
| StabRisk | (1−支持率)、重心投影から支持多角形境界までの最小距離の負値、はみ出し量 |
| WasteBuried | この配置で失われるEMS体積（将来の空隙分断） |

重み `w1..w5` はOptunaで**自作CVスコア**を最大化するようにチューニング。LB専用チューニングは禁止。

---

## 6. 高速化（4vCPU / Python）

- 高さマップ演算を **numpy スライス＋numba njit** に。最下降z・掃引干渉・支持率は全て2Dグリッドのmax/比較で表現
- 枝刈り順序は軽い判定から（内包 → 掃引 → 支持）
- **multiprocessing で並列ロールアウト**（4プロセス、GIL回避）
- 事前計算: コンテナ形状マスク、姿勢別footprint、属性テーブル
- **anytime設計**: 全モジュールに時間予算。deadline直前に必ずbestを返す

**性能目標**: 1手のfeasible判定 < 1ms、候補全列挙 < 数十ms

---

## 7. ローカルシミュレータ（最重要資産）

公式評価器を近似再現する自作環境。これの精度が順位を決める。

1. **幾何エンジン** — 内包・掃引干渉（1.5cm / 上10cm / 左右5cm）・支持率
2. **簡易定着（8cm落下）** — 支持不足時のずれを近似、閾値超で失敗フラグ
3. **PyBullet動的安定性** — 蓋あり・重力揺動でsettling、ずれと運動エネルギーを計測（開発時の検証専用、本番の8秒判断には使わない）
4. **スコア集計** — 4指標の重み付き平均

---

## 8. ロードマップ

| フェーズ | 内容 | 達成基準 |
|---|---|---|
| **0: 基盤**<br>〜1週 | 座標系・コンテナ形状のボクセル化、高さマップ、内包/最下降z/掃引/支持のnumba実装、幾何の単体テスト | feasible判定<1ms |
| **1: ベースライン**<br>2〜3週 | 課題C: EMS候補＋greedy。課題A: 元順序をそのまま返す安全提出でLB初期値取得 | **失敗ゼロ**、充填率60%台 |
| **2: 探索導入**<br>4〜7週 | 課題B: ビームサーチ＋並列ロールアウト。課題A: 焼きなまし/BRKGA。PyBullet検証ループでStabRiskを物理結果に回帰 | 充填率75〜85% |
| **3: 汎化**<br>8週〜 | Optunaで重み・解像度・ビーム幅を自作CV最大化。LB×CV相関監視。最終2枠選定 | CV分散の最小化 |

**最終2枠の選び方**: (i) LBとCVの両方で最良の攻め型、(ii) CVでスコア分散が最小の守り型。

---

## 9. 判断を変える閾値

- **自作CVとLBの相関 < 0.7** → 代理関数の重みが過学習。シーン多様化と重み再推定へ
- **greedyで実行不可が1件でも発生** → 探索より先に幾何/クリアランス実装を修正
- **オフライン3分で焼きなまし反復が数百未満** → numba化・差分評価・部分再配置を優先
- **DRLの検討**: ヒューリスティクスが頭打ち **かつ** 学習環境が安定再現できた場合のみ（PCT実装を流用）。基本は非推奨

---

## 10. 未検証の前提（着手前に潰すこと）

このブリーフの根拠は公開文献と公式資料の読解のみ。以下は**実データ・サンプルコードで確認が必要**。

| # | 前提 | 外れた場合の影響 | 確認方法 |
|---|---|---|---|
| 1 | 総合評価4指標の重み配分 | 最適化の方向自体がズレる | 公式資料の再確認、LBでの1変数実験 |
| 2 | 支持率＋支持多角形余裕が動的安定性の代理になる | 設計の柱が1本折れる | PyBulletで揺動テストし相関を測定 |
| 3 | 搬入経路＝扉法線方向への姿勢不変の並進 | 掃引判定の単純化が破綻 | 公式シミュレータの軌道定義を確認 |
| 4 | 8秒/3分で回る反復数の見積もり | ビーム幅等の推奨値が机上の空論 | 評価基盤相当のマシンで実測 |
| 5 | LD3斜めカットの寸法 | ハードコードは危険 | **公式提供の形状データを常に使う**（商用スペックには非掲載） |

---

## 11. その他のリスク

- **物理シムの再現性** — 公式エンジンと自作PyBulletは一致しない。支持率・クリアランスを公式要求より厳しめに取ってマージンで吸収
- **タイムアウト＝致命傷** — 全経路にdeadlineガードと確定解フォールバック
- **バリエーション網羅漏れ** — 棚有無・初期既積・優先専用コンテナを自作シーンに必ず含める
- **ソフト手荷物の変形モデル** — 公式の扱いとズレ得る。変形ありは上に載せない設計で回避を優先
- **AKE/AKN内寸の個体差** — 公式は形式・個体差への対応を明記。ハードコード禁止

---

## 12. 主要参考文献

**オンライン3D-BPP**
- Zhao et al., Online 3D-BPP with Constrained DRL, AAAI 2021 — https://arxiv.org/pdf/2006.14978 / code: https://github.com/alexfrom0815/Online-3D-BPP-DRL
- Zhao et al., Packing Configuration Trees, ICLR 2022 — https://openreview.net/forum?id=bfuGjlCwAq / code: https://github.com/alexfrom0815/Online-3D-BPP-PCT
- Deliberate Planning of 3D-BPP on PCT — https://arxiv.org/abs/2504.04421
- Fast Stability Validation and Stable Rearrangement — https://arxiv.org/pdf/2507.09123

**オフライン／幾何ヒューリスティクス**
- Wang & Hauser, Heightmap-Minimization, ICRA 2019 — https://ar5iv.labs.arxiv.org/html/1812.04093
- Crainic et al., Extreme Point heuristics — https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0250
- Parreño et al., Maximal-space GRASP — https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0254
- Gonçalves & Resende, BRKGA for CLP — https://www.sciencedirect.com/science/article/abs/pii/S0305054811000827
- SDF-Pack — https://arxiv.org/pdf/2307.07356

**安定性**
- Ramos et al., static mechanical equilibrium — https://www.researchgate.net/publication/220469815
- One4Many-StablePacker — https://arxiv.org/html/2510.10057v1

**ULD仕様**
- ACL Airshop — https://www.aclairshop.com/container_specs.php
- Cargo-Planner LD-3/AKE — https://cargo-planner.com/equipment-library/ld-3/ake/
- FAA AC 120-85B（貨物重心管理）— https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-85B.pdf

**手法論**
- AtCoder Heuristic Contest — https://info.atcoder.jp/overview/contest/heuristic
- Numba — https://numba.pydata.org/