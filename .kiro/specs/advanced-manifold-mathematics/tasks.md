# Implementation Plan

## 基盤準備とJIT最適化フレームワーク

- [x] 1. JIT最適化デコレータの拡張とテスト基盤構築
  - `riemannax/core/jit_decorator.py`に`@jit_optimized`デコレータを拡張
  - static_argnumsの自動検出機能を実装
  - jax.lax.condを使用した条件分岐のヘルパー関数作成
  - JITコンパイルのキャッシュ検証テスト作成
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 2. 型システムとエラーハンドリング基盤の実装
  - `riemannax/manifolds/errors.py`にManifoldError階層を実装
  - jaxtyping使用の型注釈テンプレート作成
  - 数値安定性チェックのユーティリティ関数実装
  - mypy設定の調整とpre-commitフック検証
  - _Requirements: 7.1, 7.2, 7.5_

## Grassmann多様体の数学的完全性改善

- [x] 3. SVDベースの真の指数写像実装
  - `riemannax/manifolds/grassmann.py`に`_exp_svd`メソッド実装
  - SVD分解 U·S·V^T の計算とcos(S), sin(S)の適用
  - QR分解による正規直交化の実装
  - 小さいベクトル用のlax.cond分岐実装
  - ユニットテストで可逆性検証（exp∘log = identity）
  - _Requirements: 1.1, 1.2, 1.6_

- [x] 4. SVDベースの真の対数写像とカットローカス処理
  - `_log_svd`メソッドでlog_x(y) = V·atan(S)·U^T実装
  - カットローカス検出の`_handle_cutlocus`メソッド実装
  - 2024 Handbook Algorithm 3の数値安定化適用
  - プロパティテストで数学的性質検証
  - _Requirements: 1.3, 1.4, 1.5_

- [x] 5. Grassmann多様体のバッチ処理とJIT最適化
  - vmapを使用したバッチ処理の実装
  - `test_grassmann_jit.py`でJITコンパイル検証
  - O(np²)計算複雑度のベンチマーク実装
  - パフォーマンステストで既存実装との比較
  - _Requirements: 1.6, 1.7, 6.4_

## SPD多様体の真の平行移動実装

- [x] 6. Pole ladderアルゴリズム実装（3次精度）
  - `riemannax/manifolds/spd.py`に`_pole_ladder`メソッド実装
  - n_steps=3での反復計算実装
  - 各ステップでの指数写像と対数写像の適用
  - 収束性のユニットテスト作成
  - _Requirements: 2.1, 2.3, 2.4_

- [x] 7. アフィン不変計量での閉形式平行移動
  - `_affine_invariant_transp`メソッドで閉形式解実装
  - E = Y*X^(-1)の計算とE^(1/2)の適用
  - Bures-Wasserstein計量用の特別処理実装
  - 平行移動の等長性テスト作成
  - _Requirements: 2.2, 2.5, 2.6_

- [x] 8. SPD多様体の大規模行列最適化
  - メモリ効率的なアルゴリズムの実装（n>1000）
  - Schild's ladder（1次精度）のフォールバック実装
  - `test_spd_jit.py`でJIT最適化検証
  - ベンチマークテストで性能測定
  - _Requirements: 2.4, 2.7, 6.7_

## Product Manifolds実装

- [x] 9. ProductManifoldクラスの基本構造実装
  - `riemannax/manifolds/product.py`に`ProductManifold`クラス作成
  - `__init__`でmanifolds: Tuple[BaseManifold, ...]を受け取る
  - `_split_point`と`_combine_points`メソッド実装
  - 次元計算の実装（各成分の次元の和）
  - _Requirements: 3.1, 3.5_

- [x] 10. ProductManifoldの幾何演算実装
  - 成分ごとのexp/log写像実装
  - 内積を各成分の内積の和として実装
  - 射影を成分ごとのタプルとして実装
  - 測地距離の計算実装
  - _Requirements: 3.2, 3.3, 3.4_

- [x] 11. ProductManifoldのランダムサンプリングとテスト
  - `random_point`メソッドで各成分から独立サンプリング
  - 異なる次元の多様体組み合わせテスト
  - vmapによるバッチ処理検証
  - プロパティベーステストの実装
  - _Requirements: 3.6, 3.7, 8.6_

## Quotient Manifolds実装

- [x] 12. QuotientManifoldクラスの基本構造実装
  - `riemannax/manifolds/quotient.py`に`QuotientManifold`クラス作成
  - 全空間と商構造の定義
  - `_equivalence_class_representative`メソッド実装
  - 同値類の検証テスト作成
  - _Requirements: 4.1, 4.2, 4.5_

- [x] 13. 水平リフトと水平射影の実装
  - `_horizontal_projection`メソッドで水平空間への射影
  - `_horizontal_lift`メソッドで商空間から全空間へのリフト
  - リーマン勾配の水平成分抽出実装
  - 商空間での測地距離計算
  - _Requirements: 4.3, 4.4, 4.6, 4.7_

- [x] 14. Grassmann as Stiefel/O(p)の実装例
  - Stiefel多様体を全空間としたQuotientManifold作成
  - O(p)群作用による同値関係の実装
  - 既存Grassmann実装との一致性検証
  - 最適化での同値性テスト（異なる代表元で同じ結果）
  - _Requirements: 4.2, 4.5, 8.5_

## Optimistix統合

- [x] 15. ManifoldMinimizerアダプター実装
  - `riemannax/solvers/optimistix_adapter.py`に`ManifoldMinimizer`クラス作成
  - `optx.AbstractMinimiser`を継承
  - `_convert_gradient`でEuclidean→Riemannian勾配変換
  - `_ensure_constraint`で多様体への射影実装
  - _Requirements: 5.1, 5.2, 5.5_

- [x] 16. FunctionInfoインターフェースの統合
  - Optimistixの`FunctionInfo`からの情報取得実装
  - 勾配、ヘシアン、ヤコビアンの処理
  - `AbstractManifoldSolver`基底クラスの実装
  - カスタムソルバーのテンプレート作成
  - _Requirements: 5.3, 5.6_

- [x] 17. Optimistixソルバーとの統合テスト
  - Levenberg-Marquardtソルバーでの多様体最適化テスト
  - Doglegソルバーでの制約付き最適化テスト
  - `optimistix.minimize`との互換性検証
  - `optimistix.least_squares`での多様体上最小二乗テスト
  - _Requirements: 5.4, 5.7, 8.3_

## 包括的テストとベンチマーク

- [x] 18. 数学的性質の検証テスト作成
  - `test_property_based_manifolds.py`拡張
  - 指数写像と対数写像の可逆性検証
  - 平行移動の等長性検証
  - 計量保存の検証
  - _Requirements: 8.1, 8.2, 8.5_

- [x] 19. 数値安定性とエッジケーステスト
  - 悪条件行列（条件数10^8）でのテスト
  - 境界条件と特異点のテスト
  - `test_comprehensive_numerical_stability.py`の拡張
  - カットローカス付近での挙動検証
  - _Requirements: 8.3, 8.5_

- [x] 20. パフォーマンスベンチマークと最適化
  - Pymanopt/Geooptとの比較ベンチマーク実装
  - JITコンパイル時間の測定
  - メモリ使用量のプロファイリング
  - パフォーマンス回帰検出の自動化
  - _Requirements: 6.7, 8.7, 8.4_

## ドキュメントと使用例

- [x] 21. 新機能のAPIドキュメント作成
  - 各新メソッドのdocstring（数学的背景含む）作成
  - 参照文献とアルゴリズム説明の追加
  - 型注釈の完全性確認
  - Sphinx形式でのドキュメント生成
  - _Requirements: 9.2, 9.3, 9.6_

- [x] 22. デモスクリプトとJupyter notebook作成
  - `examples/grassmann_true_exp_demo.py`作成
  - `examples/product_manifold_demo.py`作成
  - `examples/optimistix_integration_demo.py`作成
  - インタラクティブな可視化の実装
  - _Requirements: 9.1, 9.4, 9.5, 9.7_

## 最終統合と品質保証

- [x] 23. Python品質憲法の完全準拠確認
  - `mypy --config-file=pyproject.toml`でエラー0確認
  - `pre-commit run --all-files`で全チェック通過
  - `ruff check . --fix --unsafe-fixes`でリント通過
  - `pytest`で全テスト合格確認
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 24. 統合テストとE2E検証
  - 新機能と既存機能の相互作用テスト
  - 複数の多様体を組み合わせた最適化テスト
  - Optimistixソルバーでの実問題解決テスト
  - パフォーマンス目標達成の最終確認
  - _Requirements: All requirements need E2E validation_
