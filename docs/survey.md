# NEDO Challenge コンテスト2（ULD自動積付）スコア改善 技術調査レポート

## TL;DR
- **最優先は「stabilityの機序調査」＝どこがどう崩れているかを一度だけ徹底観察すること。** 文献的にも、単純な支持率は動的安定性の代理指標として弱いことが繰り返し実証されており（Ramos et al. 2015、Ali et al. 2025）、機序を見ずに支持係数・質量重みを振っても効かないのは理論的に予測できる。次点はインターロック（継ぎ目ずらし）特徴量の追加と、CEMからCMA-ES／Optunaへの乗り換え。
- **線形スコア貪欲という方策クラスは残しつつ、(a) 特徴量の追加（継ぎ目ずらし・接触交互作用・上層拘束）、(b) 候補生成をEMS＋支持ポリゴン基準へ、(c) 1手ビームサーチ（幅数十）** を入れるのが、CPUのみ／1手8秒の制約下で最もコスパが高い。非線形化（LightGBMランキング）は探索で作った教師データからの模倣学習とセットでのみ推奨。
- **公式の重み配分・揺動条件・積載率閾値は公開資料には存在せず、SIGNATEログイン制ページにある。** ユーザーが推測で置いている数値は必ずコンペのルール／データ説明／フォーラムで確認すべき。公式に確認できるのは「安定性＝全体重心位置＋動的シミュレーションで評価」という枠組みのみ。

## Key Findings

### 公式仕様について（優先順位5の確認結果）
サブエージェント調査の結論として、5指標（fill/cog/stability/placement/soft）の重み・計算式、動的安定性の揺動条件（加速度・方向・周期）、積載率46%閾値の崖、提出フィードバックの内訳は、**NEDO公式PDF（nedo.go.jp/content/800052509.pdf）・特設サイト（challenge-gh.nedo.go.jp）・METI/MLIT・SIGNATE公開ページのいずれにも記載がない**。これらはSIGNATEの会員ログイン制コンペ詳細ページ（評価方法／データ／ルール／フォーラムの各タブ）と公募説明会アーカイブ資料（YouTube配信＋説明会資料PDF）にあると強く推測される。

公式に確認できた枠組み：
- NEDO特設サイトFAQ：安定性は「積み付けた手荷物全体の重心位置や、コンピュータ上での動的シミュレーションを通して評価します」。つまり stability は静的な重心＋動的シミュレーションの二本立てで、ユーザーの理解（蓋をして重力変動→収束後のずれ）と整合。
- 公式予告PDF：充填率は「1台の有効コンテナ容積に占める搭載手荷物の容量比率」、人手で9割前後が可能とされ「できるだけ高い充填率」を求める。「充填率の定義…は正式な懸賞広告において示す予定」と明記。
- 評価は「定量評価＋成果物の定性評価」の二段階。
- コンテスト期間：スクリーニング（公開）2026年7月7日〜10月19日、成果審査（非公開）2026年11〜12月。賞金は1位500万円・2位400万円・3位300万円＋審査員特別賞。

**アクション：** 残り2ヶ月の初日に、ログイン制ページの「評価方法」「データ」「ルール」「フォーラム」を精読し、ユーザーが推測で置いている全数値（重み、揺動条件、閾値）を確定させること。これが最もリスクの低い高価値作業。フォーラムに他参加者の観測が出ている可能性もある。

### 1. 動的安定性・荷崩れ対策（最重要）

**(a) 単純な支持率は動的安定性の代理として弱い、が定説。**
Ramos, Oliveira, Gonçalves & Lopes (2015, *Transportation Research Part C*「Dynamic stability metrics for the container loading problem」) は、従来の代理指標である **M1（床に直接置かれたものを除く各荷物を支える箱の平均数）・M2（横方向支持が不足する箱の割合）** が「実世界の動的安定性の条件を表現できていない（fail to translate real-world cargo conditions of dynamic stability）」と明示し、物理シミュレータ StableCargo を使って「倒れた箱の数(NFB)」「Damage Boundary Curve フラジリティ試験内の箱数(NB_DBC)」という新指標を提案、これを多重線形回帰で解析近似した「シミュレーション不要の動的安定性メトリクス」を提示した（ScienceDirect S0968090X15003459）。Ali et al. (2025, *Computers & Operations Research*「Static stability versus packing efficiency in online 3D packing」) は4種の静的安定性制約（full-base、partial-base、支持ポリゴン、静的力学平衡）を比較し、「支持ポリゴン(polygon-based)ベースの制約は full-base／partial-base より使用ビン数の点で優れる」と結論。同分野の別の実証（Silva et al.／Junqueira & Queiroz 2022）も、full-base 制約より緩い静的力学平衡・支持ポリゴンのほうが「常により高い空間利用率で静的安定性を保証できる」と報告。

**含意（ユーザーのエージェントへの落とし込み）：** 現在の score 関数は `W_SUPPORT·支持面積率` と `SUPPORT_MIN_COVER` 足切りで支持を扱っているが、これは「支持面積率」＝ full/partial-base 系の弱い指標。**「底面重心が支持接触点の凸包（支持多角形）内に入っているか、その余裕距離はどれだけか」という支持ポリゴン指標に置き換える／併用する**のが理論的な改善方向。Gao et al. (2025, Fast Stability Validation, arXiv 2507.09123) の Load-Bearable Convex Polygon (LBCP、質量分布を知らなくても崩壊しない支持位置を近似定数時間で判定)、O4M-SP (arXiv 2510.10057) の「新規荷物の底面幾何中心が接触面の凸包内にあること＝support constraint」が実装リファレンスになる。

**(b) インターロック（継ぎ目ずらし）は動的安定性を上げる。**
実務のパレタイジング文献は一致して、column stacking（柱積み）は圧縮強度は高いが荷崩れに弱く、interlocking（煉瓦積み、層ごとに90度回転／継ぎ目ずらし）が横方向安定性を上げるとする。Robotiq のガイド（"Palletizing Pallet Pattern Charts Explained"）は定量的に、**柱積み(column stacking)は箱の安定性を25–30%高めるが「構造自体は安定性を提供せず、ラッシングやシュリンクラップに依存する」**とし、**部分インターロックは「完全にインターロックした構造と比べたとき、各鉛直層の強度を最大45%高める（improves the strength of each vertically aligned layer by up to 45%）」**と整理している（blog.robotiq.com/palletizing-pallet-pattern-charts）。Packaging World は「下層は圧縮強度のため柱積み、上層は安定性のためインターロック」というハイブリッドを推奨。IATA/ULD実務（uldcare.com「Build stable loads」）も「パレット上の貨物は可能ならインターロック状に積む」と明記。

**含意：** 現 score 関数の `− W_SEAM·縁の揃い` は「縁が揃うほど減点」＝インターロックを弱く促す方向だが、これは継ぎ目の"連続性"を評価しているだけ。**「直下の荷物の鉛直継ぎ目（垂直な接触面境界）と、今置く荷物の継ぎ目がどれだけズレているか」を報酬にする継ぎ目ずらし特徴量 `W_STAGGER·継ぎ目ずれ量` を新設**するのが直接的。これは航空手荷物のような不定形・多サイズ荷物では特に効く（同サイズ箱前提のパターン生成は不要で、隣接箱との境界オフセットのみ見ればよい）。

**(c) 隙間の充填と壁接触が横加速度に効く。**
EUMOS 40509（欧州の荷崩れ試験規格、2008年にKU LeuvenのMarc Juwetが考案、EU Directive 2014/47/EU の Annex 3 が参照）は、テストパレットを台上で加速する動的試験で、**加速度を50ms以内に到達させ0.3秒以上一定に維持する**（Wikipedia "Acceleration test":「accelerated at a constant acceleration for 0.3 seconds or longer. This constant acceleration level must be reached within 50ms」）。参照加速度は EN 12195-1 で**前方約0.8g、横・後方0.5g**、EUMOS 40509 は最大0.8gを300ms以上維持（packcalc.com）。認証基準は**最大弾性変形10%以下・最大恒久変形5%以下・包装破損なし**（Wikipedia）。Wikipedia は「0.5gの加速度試験に耐える荷は、単一のラッシング、車両壁による支持、または他の荷による支持で法定安全レベルを満たせる」とする。つまり **壁接触・隣接荷物による相互支持・隙間の少なさ**が横加速度耐性の主要因子で、これは現 score の `W_WALL·接する壁の数` の方向性が正しいことを裏づける（重みを上げる価値あり）。内部空隙が動く余地を作るという仮説も、変形・ずれを厳しく制限するEUMOS基準の思想と整合的。

**(d) 天井（蓋）直下の拘束。**
ULD実務ではネット／蓋が最上層を上から押さえる。ユーザーの評価も「蓋をして」動的変動をかける。最上層が蓋に接していないと拘束されず自由に動ける→崩れやすい。**最上層の荷物と天井のギャップが小さいほど加点する特徴量、または最上層だけ `W_CEIL`（上の隙間ペナルティ）を強める**のが理にかなう。

**(e) 2023–2026の最新研究の位置づけ。**
- **One4Many-StablePacker** (arXiv 2510.10057)：報酬に「loading rate ＋ 新規のheight difference（高さ差）メトリクス」を統合し、より平ら（flat）な配置を促す。これはユーザーの課題B「上に積める平らな面が残らない」に直接対応する知見。support制約（底面中心が接触凸包内）とweight制約（積載荷重が耐荷重比 rw 以内）を明示的に課す。
- **Fast Stability Validation** (arXiv 2507.09123)：LBCPで質量分布を知らなくても崩壊しない支持位置を判定、さらに既存荷物を再配置して新荷物を収める Stable Rearrangement Planning を提案。
- **SDF-Pack** (arXiv 2307.07356)：truncated SDF で「既存荷物とどれだけ密に接するか」をコンパクトネス指標化。現 score の `W_WASTE`（足元の空洞）や `W_CEIL` を SDF 的な密着度で置き換える発想の源。
- **RoboBPP** (arXiv 2512.04415) / **PCT deliberate planning** (arXiv 2504.04421)：テスト時の物理シミュレーション回数 k_d を増やすほど実搬送安定性が上がる（quasi-static 55% → k_d=8 で100%）と報告。訓練は高速な代理でよいがテスト時シミュレーションは必須、との結論。ユーザーのエージェントが持つ ROLLOUT_SETTLE を「本命候補に対してだけ回数を増やす」方向のヒント。

### 2. 積付戦略・候補生成（課題B：置き場所が尽きる）

**候補生成方式の比較：** Corner Point (Martello 2000)、Extreme Point (Crainic 2008)、Empty Maximal Space (Ha 2017) が三大手法。PCT論文の実験では、これら適切な展開スキームに導かれた方策は全座標(FC)空間より一貫して優れる。EMSは「その隅から軸方向にこれ以上広げられない最大の空箱」で、実装が単純かつ有効。GOPT (arXiv 2409.05344) は heightmap のX/Y方向の高さ変化から角点を検出しEMSを生成する高速実装を提示。近年の OPAL (arXiv 2607.28257) は EMS 生成を「低く・よく支持され・コンパクトで空間的に多様な配置」を優先するよう運用ガイドする OG-EMS を提案しており、ユーザーの課題（低重心・支持・平坦の同時達成）に方向性が近い。

**「平らな面が残らない」問題への対策：**
- Wang & Hauser (2019, ICRA) の **Heightmap-Minimization**：配置後の占有体積増加が最小になる置き方を選ぶ＝表面を平らに保つ。ユーザーの課題Bの核心（重心0.72m対し最上面1.48mで天井まで8cm、内部空隙0.4–1.3m³）に直撃する。
- O4M-SPのheight difference報酬も同趣旨。**現 score に「配置後の高さマップの粗さ（分散／最大高さの増分）を減らす」項 `− W_FLAT_SURF·高さマップ粗さ増分` を追加**すべき。現在の `W_FLAT·縦寸法`（背の高い荷物を避ける）は代理として弱く、置いた後の表面凹凸を直接見るほうがよい。

**搬入経路制約（手前から直線押し込み、1.5cmクリアランス）：**
Wang & Hauser (RSS 2019「Robot Packing with Known Items and Nondeterministic Arrival Order」) は、積載方向に沿ったクリアランス制約（押し込む箱の面から伸びる prism 上に既存荷物がないこと）と、マニピュレータの搬入経路の干渉を明示的に定式化。ULD実務の定石は「奥から手前へ、下から上へ、階段状に」。**候補生成の段階で「その候補へ手前から直線押し込みしたとき既存荷物と干渉するか」を先に判定し、干渉する候補を除外**すれば、検証NGでの早期終了（transportで死ぬ）を減らせる。現在は検証で初めて弾かれている可能性が高い。

**棚ありコンテナ（課題C）：** 棚下は「後から入れにくい空間」。棚下候補の無条件追加は soft を約10下げると実証済み（ユーザー観測）。→ **棚下は「早い段階で・ハード荷物・小型のものだけ」で埋める順序制御**が正着。オフライン順序最適化がある課題Aでは、棚下充填用にハード小型荷物を順序の前方に寄せる。オンラインのみの課題B/Cでは、棚下候補を出すのはプール内にハード小型がある時だけに限定するゲートを入れる。

**不定形・ソフトバッグ：** 変形前提のpackingは研究が少なく、実務では「ソフトは上／隙間、下敷きにしない」がルール（公式③と一致）。現 score の `− W_SOFT_DEFER·[ソフト]` の方向は正しい。

### 3. 限られた計算資源での探索（4 vCPU / Python / 1手8秒・オフライン180秒）

**ビームサーチが本命。** AHC上位陣の定石（terry-u16 の解説群、thunder『ゲームで学ぶ探索アルゴリズム実践入門』、chokudaiサーチ）では、配置系の逐次決定問題はビームサーチが強い。1手8秒あれば、幅（beam width）数十〜百で、各ノードで候補（位置×向き）を線形scoreで評価し上位を残す。**差分評価が鍵**：高さマップをnumpy配列で持ち、1手で変わる領域だけ更新（AHC参加記でも「差分計算の実装で数倍高速化」が定番の勝ち筋）。物理ロールアウト（定着確認）は重いので、ビーム内では線形scoreによる近似評価のみ、最終的に選ぶ1手だけ物理検証、という二段構えにする。

**軽量化の具体策：**
- 高さマップの衝突判定・支持面積計算を numpy ベクトル化、ホットループは numba の `@njit` 化。
- multiprocessing で候補評価を4 vCPUに分散（1手8秒・オフライン180秒とも並列で稼げる）。
- 物理シミュレーション（PyBullet）呼び出しは最小化。RoboBPP的に「本命候補のみ物理、それ以外は幾何＋支持ポリゴンの高速代理」。

**オフライン180秒（課題A）：** 順序最適化に焼きなまし／ビームサーチを使える。近傍は「2手荷物のswap」「1荷物の順序移動」。評価は軽量な代理関数（充填率＋高さマップ粗さ＋支持ポリゴン余裕＋継ぎ目ずれ）で回し、時間管理は経過時間ベースで温度を下げる（terry-u16らの定石）。AHCの知見として、評価関数が平坦だと解がブラウン運動して焼けないため、生スコアに補助項（連続的な勾配を与える項）を足すと収束が改善する。

### 4. 重み・パラメータ最適化（CEMが頭打ちした問題）

**CEMが早期収束した原因は典型的。** CEMは共分散が速く縮退し、局所解に premature convergence する（CEM-RL arXiv 1810.01222 は「premature convergenceを防ぐため追加分散εを足す」と明示；CMA-ES解説群も「CEMは共分散が速く退化し、最適点付近に十分な点を撒けず premature convergence する」と指摘）。ユーザーの症状（世代最良値が横ばい、第7世代1位2位差0.06で出荷個体はほぼ偶然）はまさにこれ。加えて**評価ノイズが大きい**（PyBulletエピソード1本、シーン依存）ため、少数エリートの平均が偶然に振られる。

**推奨対策（順に）：**
1. **CMA-ESへ乗り換え。** Hansen のチュートリアル(arXiv 1604.00772)通り、CMA-ESはステップサイズと共分散を適応させ premature convergence を回避。Python では `cmaes` ライブラリ(arXiv 2402.01373)が軽量。ノイズ対策として `cm<1`（平均更新の縮小、Hansen が「ノイズ関数では有利」と明記）が有効。
2. **共通乱数（common random numbers）＋固定シーンセット。** 全個体を同じシーン集合・同じ物理シードで評価し、個体間比較の分散を消す。これは評価ノイズ下で「本当に良くなったか」を判定する最重要テクニック。
3. **評価シーンの是正。** ユーザーの致命的バグ：訓練64シーン中、実configと同じオフライン条件(optimize=True, look_ahead=1)が1シーンしかなかった。**評価シーン分布を本番3パターン（課題A/B/C）に合わせて再構成**しないと、何を最適化しても本番に効かない。
4. **エージェント本体を固定してから最適化。** CEM後にOFFLINE_TIEBREAK/ROLLOUT_SETTLEを変えたため重みが旧エージェント最適。**コード凍結→最適化の順序を厳守。**
5. **Optuna (TPE)** も選択肢。多忠実度（low-fidelity＝物理を粗く／シーン数を減らして粗選別→有望個体だけ高忠実度）でサンプル効率を上げる（多忠実度BOの定石。低忠実で609候補中の最適解を低忠実30＋高忠実7評価で特定した事例もある）。1評価が重い状況に適する。

**線形スコア方策クラスの限界と非線形化：**
14特徴の線形和は表現力に限界（特徴の交互作用＝「支持が低い×重い」等を表せない）。選択肢：
- **交互作用項を手で足す**（例：`質量×底面高さ` は既にある。`支持ポリゴン余裕×高さ`、`継ぎ目ずれ×層番号` 等を追加）。最も安全でコストが低い。
- **LightGBM/XGBoostのlearning-to-rank（LambdaMART）** で候補ランキングを学習。CPU推論は軽く、1手8秒に収まる。ただし**教師データが必要**→ビームサーチ等の探索で得た良解の「選ばれた候補 vs 選ばれなかった候補」をランキング教師にする（模倣学習）。
- **小さなMLP**も可能だがGPUなしでの学習コストと過学習リスクを考えると、まずGBDTランカーが現実的。

**模倣学習（探索→貪欲方策）：** behavior cloning の定石通り、重いビームサーチ／MCTSを「教師」に、その選択を線形／GBDTスコアで再現するよう学習させれば、本番の1手8秒制約下で教師に近い質を高速に出せる。単純なBCは分布シフト（誤差の累積）に弱いため、DAgger／forward training で学習方策が訪れた状態にも教師ラベルを与えて補正するのが有効（空港地上業務VRPの LNS で forward training が compounding error を補正した事例 arXiv 2302.13797 が構造的に近い）。

### 補助：過去コンペ・実務知見
- IATA ULD Regulations / uldcare.com：ULDは航空機部品扱い、重量・重心制限厳守、ネットで拘束、可能ならインターロック積み。NCA「Shipper Built Unit Guidelines」：小型カートンはインターロック積み、パイル缶は2層まで（超える場合はスプレッダーボード＋シュリンクラップ）等の実務ルール。
- StableLego (arXiv 2402.10711)：ブロック積みの力学的安定性解析。継ぎ目とインターロック接続の力学モデルは「パレタイジング等の通常ブロック積みにも、インターロック接続由来の候補力を除けば拡張できる」と明記。

## Details（提案別：期待効果・実装コスト・根拠の確かさ）

| # | 提案 | 期待効果 | 実装コスト | 根拠の確かさ |
|---|------|---------|-----------|-------------|
| A | **stability機序の観察**（崩れが上層/下層か、塔/壁か、初手か終盤か。物理ログを可視化） | 直接効果はないが以降の全施策の的中率を左右。stability 22→60で合成値+7.6（ユーザー試算） | 低（数日、既存PyBulletログ） | 高（Ramos, Aliが単純支持率の不十分さを実証） |
| B | 支持面積率→**支持ポリゴン余裕**へ置換／併用 | stability中〜大 | 中（凸包計算） | 高（O4M-SP, LBCP, Ali 2025） |
| C | **継ぎ目ずらし特徴量** `W_STAGGER` 追加、`W_SEAM`の符号見直し | stability大（横加速度耐性） | 低（隣接境界のオフセット計算） | 中〜高（パレタイジング実務、IATA、StableLego） |
| D | **高さマップ粗さ最小化**項の追加（Heightmap-Minimization） | fill＋課題Bの詰まり解消 | 中 | 高（Wang&Hauser, O4M-SP height diff） |
| E | 候補生成を**EMS＋搬入経路事前判定**へ | 課題Bのtransport死を減、fill | 中〜高 | 高（PCT, GOPT, OPAL, Wang&Hauser RSS） |
| F | **最上層の天井ギャップ拘束**項 | stability中 | 低 | 中（ULD実務、蓋評価の物理） |
| G | 棚下を**ハード小型限定・早期**で埋めるゲート | 課題C積載率、soft維持 | 中 | 中（実務＋ユーザー実測） |
| H | **1手ビームサーチ**（幅数十、差分評価、本命のみ物理） | 全課題の質、特にB/C | 中〜高 | 高（AHC定石、RoboBPP） |
| I | CEM→**CMA-ES＋共通乱数＋シーン是正** | 重み最適化が再び効く | 中 | 高（Hansen, CEM-RL） |
| J | **GBDTランカー＋模倣学習**（Hから教師生成） | 非線形化の上積み | 高 | 中（LTR定説だが本タスク未実証） |

## Recommendations（残り2ヶ月・着手順）

**第0週（最初の数日）：土台固め。**
1. SIGNATEログイン制ページで公式の重み・揺動条件・積載率閾値・フィードバック仕様を確定（推測を排除）。フォーラムも確認。
2. **stabilityの機序観察（提案A）。** 物理シミュレーション後のフレームを保存し、「どの荷物が最初に・どれだけ動くか」「上層か下層か」「塔状か壁際か」を数エピソード目視＋数値集計。**これを最初にやるべきかは文献的にも明確にYES**：Ramos(2015)/Ali(2025)が「単純支持率は動的安定性の代理として不適」と実証しており、機序を見ずに支持係数を振るのはユーザーの実績（機序調査2勝、パラメータ0勝23敗）とも完全に一致する。**"なぜ崩れるか"を1回見れば、B/C/Fのどれを最優先すべきかが決まる。**

**第1–3週：stability直撃施策。** 観察結果に応じて提案B（支持ポリゴン）、C（継ぎ目ずらし）、F（天井拘束）を1つずつ導入。各導入は共通乱数＋固定シーンでA/B比較し、ノイズに埋もれない改善量（例：合成値+1以上）だけ採用。1施策=1変更を厳守（機序調査型の勝ちパターン）。

**第3–5週：課題B/Cの詰まり解消。** 提案D（高さマップ粗さ）とE（EMS＋搬入経路事前判定）。課題BのtransportでのDoA率72–94%が最大の伸びしろ。G（棚下ゲート）も。

**第5–7週：探索と再最適化。** 提案H（1手ビームサーチ）を差分評価で軽く入れる。**エージェント本体を凍結**した上で提案I（CMA-ES＋共通乱数、本番3パターンに合わせたシーンで）を回す。

**第7–8週：仕上げ。** 余力があれば提案J（GBDTランカー＋模倣学習）。時間切れなら見送り、暫定/最終でシーンが変わる点を踏まえ**過学習を避けた頑健な重み**で2ファイル提出。

**判断を変える閾値：**
- 機序観察で「崩れが下層の初期配置起因」なら B/C を最優先、「上層が蓋直下で暴れる」なら F を最優先。
- CMA-ESでも共通乱数下で改善が+0.5未満なら、方策クラスの限界＝J（非線形化）へ資源を移す。
- 課題Bのtransport死が E 導入後も50%超なら、候補生成ではなく「置ける面を作る」＝D（平坦化）を強化。

## Caveats
- **公式の重み・揺動条件・積載率閾値・フィードバック仕様は公開資料に存在せず、本レポートはユーザー提供の推測仕様に基づく部分がある。** 必ずログイン制ページで確定すること。特にstabilityの揺動条件（0.3g×4方向×2周期は推測）が実際と違えば、継ぎ目ずらし/壁接触/天井拘束の相対的重要度が変わる。
- EUMOS 40509（前方0.8g／横0.5g、EN 12195-1）は道路輸送の規格であり、航空ULDの実際の揺動条件とは異なる。方向性（横加速度に何が効くか）の参考にとどめ、数値をそのまま流用しない。なお本文中の「層間ずれ2%以内」は今回の調査で一次規格の直接根拠を確認できなかった（EUMOSの確認済み公式変形基準は弾性変形10%以下・恒久変形5%以下）。
- stabilityの学術的代理指標（支持ポリゴン、LBCP、Ramos回帰メトリクス）は「本番の動的シミュレーション」と完全一致する保証はない。最終判定は必ず本番相当の物理検証で行うこと（RoboBPP：訓練は代理でよいがテスト時シミュレーションは必須）。
- Robotiqの「柱積み＋25–30%」「部分インターロック＋45%」は業者ガイドの数値であり、査読論文ではない。方向性の根拠として扱い、絶対値は鵜呑みにしない。
- GBDTランカー＋模倣学習（提案J）は本タスクでの実証例がなく、教師データ生成（ビームサーチ）が前提。時間対効果が読みにくく、最後に回すべき。
- 非線形化・ビームサーチはいずれもCPU・8秒制約に収まる設計が必須。numba/numpy差分評価を怠ると時間超過→ランダム行動で逆効果。

---
### 主な参照文献・URL
- One4Many-StablePacker (O4M-SP): https://arxiv.org/abs/2510.10057
- Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement (LBCP): https://arxiv.org/pdf/2507.09123
- SDF-Pack: https://arxiv.org/abs/2307.07356 ／ コード: https://github.com/kwpoon/SDF-Pack
- Ramos et al. "Dynamic stability metrics for the container loading problem" (TR-C 2015): https://www.sciencedirect.com/science/article/abs/pii/S0968090X15003459
- Ramos et al. "A container loading algorithm with static mechanical equilibrium stability constraints" (TR-B 2016): https://www.sciencedirect.com/science/article/abs/pii/S0191261515302022
- Ali et al. "Static stability versus packing efficiency in online 3D packing" (C&OR 2025): https://www.sciencedirect.com/science/article/abs/pii/S0305054825000334
- Wang & Hauser "Stable Bin Packing of Non-convex 3D Objects" (ICRA 2019, Heightmap-Minimization): https://arxiv.org/pdf/1812.04093
- Wang & Hauser "Robot Packing with Known Items and Nondeterministic Arrival Order" (RSS 2019): https://www.roboticsproceedings.org/rss15/p35.pdf
- Deliberate Planning of 3D Bin Packing on PCT: https://arxiv.org/pdf/2504.04421
- RoboBPP (物理ベースベンチマーク): https://arxiv.org/html/2512.04415
- GOPT (Transformer + EMS): https://arxiv.org/pdf/2409.05344
- OPAL (OG-EMS): https://arxiv.org/html/2607.28257
- StableLego: https://arxiv.org/pdf/2402.10711
- Junqueira & Queiroz「support factor の静的安定性回帰分析」: https://onlinelibrary.wiley.com/doi/epdf/10.1111/itor.12750
- パレタイジングパターン（柱積み/インターロック定量）Robotiq: https://blog.robotiq.com/palletizing-pallet-pattern-charts
- 荷崩れとパターン Packaging World: https://www.packworld.com/home/article/13372817/reducing-the-occurrence-of-collapsing-pallet-loads-part-ii
- EUMOS 40509 / Acceleration test: https://en.wikipedia.org/wiki/Acceleration_test ／ https://packcalc.com/resources/pallet-load-stability-physics-tilt-angles
- IATA ULD Regulations: https://www.iata.org/en/publications/manuals/uld-regulations/ ／ 実務: https://www.uldcare.com/uld-care-code-conduct/build-stable-loads-observe-aircraft-uld-limitations/
- CEM-RL: https://arxiv.org/pdf/1810.01222 ／ CMA-ES tutorial (Hansen): https://arxiv.org/pdf/1604.00772 ／ cmaes ライブラリ: https://arxiv.org/pdf/2402.01373
- 学習型3D-BPP文献レビュー: https://arxiv.org/html/2312.08103v1
- Extreme Point heuristics (Crainic 2008): https://www.cirrelt.ca/documentstravail/cirrelt-2007-41.pdf
- 模倣学習/forward training（空港VRP LNS）: https://arxiv.org/pdf/2302.13797
- LightGBM learning-to-rank (LambdaMART): https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- AHC探索テクニック（ビームサーチ/焼きなまし）: https://www.terry-u16.net/entry/ahc-practice-problem ／ https://jetbead.github.io/AtCoderHeuristicContestMemo/Library/beam_search.html
- NEDO公式（コンテスト概要PDF）: https://www.nedo.go.jp/content/800052509.pdf ／ 特設サイト: https://www.challenge-gh.nedo.go.jp/ ／ SIGNATE: https://service.signate.jp/campaign/baggage-loading-robot-2026