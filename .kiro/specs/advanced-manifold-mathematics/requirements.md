# Requirements Document

## Introduction

RiemannAXの数学的完全性と機能性を大幅に向上させる包括的なエンハンスメントです。本機能は、リーマン多様体最適化ライブラリとしての理論的完全性を確立し、新たな多様体構造のサポートを追加し、Optimistixとの深い統合により制約付き最適化を実現します。これにより、RiemannAXは学術研究および産業応用の両方において、世界標準のリーマン幾何学ライブラリとしての地位を確立します。

## Requirements

### Requirement 1: Grassmann多様体の数学的完全性改善
**User Story:** 研究者として、数学的に正確な指数写像と対数写像を持つGrassmann多様体を使用し、理論的に保証された最適化を実行したい

#### Acceptance Criteria

1. WHEN Grassmann多様体で`exp`メソッドが呼び出される THEN システムはSVDベースの真の指数写像アルゴリズムを実行 SHALL する
2. IF 接ベクトルξのSVD分解がU·S·V^Tである THEN 指数写像は`z = p·V·cos(S)·V^H + U·sin(S)·V^H`を計算してQR分解 SHALL する
3. WHEN Grassmann多様体で`log`メソッドが呼び出される THEN システムは`log_x(y) = V·atan(S)·U^T`の公式を使用 SHALL する
4. WHERE 数値的安定性が必要な場合 THE システムは修正アルゴリズム（2024 Handbook Algorithm 3）を適用 SHALL する
5. WHEN カットローカス上の点が検出される THEN システムは非一意なマッピングを適切に処理 SHALL する
6. WHILE 計算が実行される THE システムはO(np²)の計算複雑度を維持 SHALL する
7. IF バッチ処理が要求される THEN システムはvmapによる効率的なベクトル化を提供 SHALL する

### Requirement 2: SPD多様体の真の平行移動実装
**User Story:** 機械学習エンジニアとして、数学的に正確な平行移動を持つSPD多様体を使用し、共分散行列の最適化において理論的保証を得たい

#### Acceptance Criteria

1. WHEN SPD多様体で`transp`メソッドが呼び出される THEN システムは真の平行移動アルゴリズムを実行 SHALL する
2. IF アフィン不変計量が使用される THEN システムは閉形式の平行移動公式を適用 SHALL する
3. WHERE 閉形式解が利用できない場合 THE システムはPole ladderアルゴリズム（3次精度）を使用 SHALL する
4. WHEN Schild's ladderが要求される AND 比較が必要な場合 THEN システムは1次精度のSchild's ladder実装も提供 SHALL する
5. IF 可換行列が検出される THEN システムはBures-Wasserstein計量用の特別な公式を適用 SHALL する
6. WHILE 平行移動が計算される THE システムはアフィン不変性を保証 SHALL する
7. WHEN 大規模行列（n>1000）が処理される THEN システムはメモリ効率的なアルゴリズムを使用 SHALL する

### Requirement 3: Product Manifolds実装
**User Story:** 最適化研究者として、複数の多様体の直積上で最適化を実行し、複雑な制約を持つ問題を解決したい

#### Acceptance Criteria

1. WHEN `ProductManifold`クラスがインスタンス化される THEN システムは複数の基底多様体のリストを受け入れ SHALL する
2. IF Product多様体で`exp`が呼び出される THEN システムは各成分多様体で個別にexp mapを適用 SHALL する
3. WHERE 内積計算が必要な場合 THE システムは各成分の内積の和を返す SHALL する
4. WHEN `proj`メソッドが呼び出される THEN システムは各成分への射影をタプルとして返す SHALL する
5. IF 異なる次元の多様体が組み合わされる THEN システムは適切な次元管理を提供 SHALL する
6. WHILE 最適化が実行される THE システムは各成分多様体の制約を同時に満たす SHALL する
7. WHEN `random_point`が呼び出される THEN システムは各成分から独立にサンプリング SHALL する

### Requirement 4: Quotient Manifolds実装
**User Story:** 幾何学研究者として、商多様体上での最適化を実行し、対称性を持つ問題を効率的に解決したい

#### Acceptance Criteria

1. WHEN `QuotientManifold`クラスが作成される THEN システムは全空間と同値関係を定義 SHALL する
2. IF Grassmann多様体がStiefel多様体の商として要求される THEN システムは適切な同値類を実装 SHALL する
3. WHERE 水平リフトが必要な場合 THE システムは商空間から全空間への適切なリフティングを提供 SHALL する
4. WHEN 最適化が商多様体上で実行される THEN システムは代表元上での最適化として実装 SHALL する
5. IF 同値類の異なる代表元が与えられる THEN システムは同じ最適化結果を保証 SHALL する
6. WHILE リーマン勾配が計算される THE システムは水平成分のみを抽出 SHALL する
7. WHEN `dist`メソッドが呼び出される THEN システムは商空間での正しい測地距離を計算 SHALL する

### Requirement 5: Optimistixとの制約付き最適化統合
**User Story:** データサイエンティストとして、リーマン多様体制約を持つ最適化問題をOptimistixの高度なソルバーで解決したい

#### Acceptance Criteria

1. WHEN RiemannAX多様体がOptimistixソルバーに渡される THEN システムはシームレスな統合を提供 SHALL する
2. IF `optimistix.minimize`が多様体制約付きで呼び出される THEN システムは自動的にリーマン勾配に変換 SHALL する
3. WHERE カスタムソルバーが必要な場合 THE システムは`AbstractManifoldSolver`基底クラスを提供 SHALL する
4. WHEN Levenberg-Marquardtソルバーが使用される THEN システムは多様体上での適切なダンピングを実装 SHALL する
5. IF 制約違反が検出される THEN システムは自動的に多様体への射影を実行 SHALL する
6. WHILE 最適化が進行する THE システムはOptimistixの収束基準とコールバックをサポート SHALL する
7. WHEN `optx.least_squares`が多様体上で呼び出される THEN システムは適切なJacobian変換を提供 SHALL する

### Requirement 6: JAX JIT最適化とパフォーマンス
**User Story:** MLエンジニアとして、すべての新機能がJAX JITで完全に最適化され、高速な実行を実現したい

#### Acceptance Criteria

1. WHEN 新しい多様体メソッドが実装される THEN すべてのメソッドは`@jit_optimized`デコレータを持つ SHALL する
2. IF static引数が存在する THEN システムは適切な`static_argnums`を指定 SHALL する
3. WHERE 条件分岐が必要な場合 THE システムは`jax.lax.cond`を使用 SHALL する
4. WHEN vmapが適用される THEN システムはバッチ次元を正しく処理 SHALL する
5. IF メモリ効率が重要な場合 THEN システムはインプレース操作を最大限活用 SHALL する
6. WHILE JITコンパイルが実行される THE システムは初回コンパイル時間を最小化 SHALL する
7. WHEN パフォーマンステストが実行される THEN 新機能は既存実装と同等以上の速度を達成 SHALL する

### Requirement 7: 型安全性とPython品質憲法準拠
**User Story:** 開発者として、すべての新コードが完全な型安全性を持ち、品質基準を満たすことを保証したい

#### Acceptance Criteria

1. WHEN 新しいコードがコミットされる前 THEN `mypy --config-file=pyproject.toml`がエラーなしで通過 SHALL する
2. IF pre-commitフックが実行される THEN `pre-commit run --all-files`が完全に成功 SHALL する
3. WHERE lintチェックが必要な場合 THE `ruff check . --fix --unsafe-fixes`がエラーなしで完了 SHALL する
4. WHEN テストが実行される THEN `pytest`がすべてのテストで成功 SHALL する
5. IF 型注釈が追加される THEN システムはPython 3.10+の最新型システムを使用 SHALL する
6. WHILE 開発が進行する THE システムは技術的負債の蓄積を防止 SHALL する
7. WHEN ドキュメントが生成される THEN すべての公開APIは完全な型注釈を含む SHALL する

### Requirement 8: 包括的テストカバレッジ
**User Story:** QAエンジニアとして、新機能が既存機能を破壊せず、数学的正確性が検証されることを保証したい

#### Acceptance Criteria

1. WHEN 新しい多様体が実装される THEN 対応する`test_[manifold].py`と`test_[manifold]_jit.py`が作成される SHALL する
2. IF 数学的性質がテストされる THEN システムは理論的性質（可逆性、計量保存等）を検証 SHALL する
3. WHERE 数値安定性が重要な場合 THE システムは悪条件行列でのテストを含む SHALL する
4. WHEN 統合テストが実行される THEN 新機能と既存機能の相互作用が検証される SHALL する
5. IF エッジケースが存在する THEN システムは境界条件と特異点を適切にテスト SHALL する
6. WHILE プロパティベーステストが実行される THE システムは数学的不変量を検証 SHALL する
7. WHEN パフォーマンスベンチマークが実行される THEN システムは回帰を検出して報告 SHALL する

### Requirement 9: ドキュメントと例
**User Story:** ユーザーとして、新機能の使い方を理解し、実際の問題に適用するための明確な例が欲しい

#### Acceptance Criteria

1. WHEN 新しい多様体が追加される THEN 対応する`examples/[manifold]_demo.py`が作成される SHALL する
2. IF APIドキュメントが生成される THEN すべてのメソッドは数学的背景を含むdocstringを持つ SHALL する
3. WHERE 複雑なアルゴリズムが実装される THE システムは参照文献とアルゴリズムの説明を提供 SHALL する
4. WHEN Jupyter notebookが作成される THEN インタラクティブな可視化と説明を含む SHALL する
5. IF Optimistix統合が使用される THEN システムは統合の具体例を提供 SHALL する
6. WHILE ドキュメントが維持される THE システムはコードとドキュメントの同期を保証 SHALL する
7. WHEN ユーザーガイドが更新される THEN 新機能のベストプラクティスが含まれる SHALL する
