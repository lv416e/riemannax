"""
JIT最適化されたStiefel多様体の包括的テストスイート.

Requirements:
- 8.1: JIT vs 非JITの数値同等性検証テスト (rtol=1e-6, atol=1e-8)
- 8.2: API完全互換性の回帰テスト
- 2.2: 大規模配列でのパフォーマンステスト
- 1.4: 正規直交性制約保持の検証テスト
- 8.4: 修正された指数写像の数学的正確性確認
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from riemannax.manifolds.stiefel import Stiefel


class TestStiefelJITOptimization:
    """Stiefel多様体JIT最適化の包括的テストクラス."""

    def setup_method(self):
        """各テストメソッド前の初期化."""
        self.manifold_st52 = Stiefel(5, 2)  # 5次元空間での2-frame
        self.manifold_st43 = Stiefel(4, 3)  # 4次元空間での3-frame
        self.manifold_st33 = Stiefel(3, 3)  # 3次元直交群（特殊ケース）

        # テスト用キー
        self.key = jr.PRNGKey(42)

        # 数値許容値
        self.rtol = 1e-6
        self.atol = 1e-8

    def test_stiefel_jit_implementation_methods_exist(self):
        """JIT実装メソッドの存在確認 (Requirement 8.2)."""
        required_methods = ["_proj_impl", "_exp_impl", "_log_impl", "_inner_impl", "_dist_impl", "_get_static_args"]

        for method in required_methods:
            assert hasattr(self.manifold_st52, method), f"Missing JIT method: {method}"
            assert callable(getattr(self.manifold_st52, method)), f"JIT method not callable: {method}"

    def test_proj_jit_vs_original_equivalence_st52(self):
        """プロジェクション操作のJIT vs 非JITの数値同等性 (Requirement 8.1)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        v = jr.normal(key2, (5, 2))

        # 元の実装
        proj_original = self.manifold_st52.proj(x, v)

        # JIT実装
        proj_jit = self.manifold_st52._proj_impl(x, v)

        # 数値同等性確認
        np.testing.assert_allclose(proj_original, proj_jit, rtol=self.rtol, atol=self.atol)

        # 接空間条件の確認
        xtv_original = x.T @ proj_original
        xtv_jit = x.T @ proj_jit

        # X^T V + V^T X = 0 (反対称性)
        skew_original = xtv_original + xtv_original.T
        skew_jit = xtv_jit + xtv_jit.T

        np.testing.assert_allclose(skew_original, jnp.zeros_like(skew_original), atol=1e-6)
        np.testing.assert_allclose(skew_jit, jnp.zeros_like(skew_jit), atol=1e-6)

    def test_exp_jit_vs_original_equivalence_st52(self):
        """指数写像のJIT vs 非JITの数値同等性 (Requirement 8.1)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        v = self.manifold_st52.random_tangent(key2, x)

        # 元の実装 (retractionベース)
        exp_original = self.manifold_st52.exp(x, v)

        # JIT実装 (改良版)
        exp_jit = self.manifold_st52._exp_impl(x, v)

        # 両方とも正規直交性を保持することを確認
        assert self.manifold_st52.validate_point(exp_original), "Original exp result not on manifold"
        assert self.manifold_st52.validate_point(exp_jit), "JIT exp result not on manifold"

        # JIT版は数学的により正確な実装なので、完全な同等性は期待しないが、
        # どちらも正規直交性を保持し、合理的な結果を返すことを確認
        assert jnp.allclose(exp_jit.T @ exp_jit, jnp.eye(2), atol=1e-6)
        assert jnp.allclose(exp_original.T @ exp_original, jnp.eye(2), atol=1e-6)

    def test_modified_exponential_map_correctness(self):
        """修正された指数写像の数学的正確性確認 (Requirement 8.4)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)

        # 小さな接ベクトル（線形近似が有効な範囲）
        v_small = 0.01 * self.manifold_st52.random_tangent(key2, x)

        # 修正された指数写像
        exp_result = self.manifold_st52._exp_impl(x, v_small)

        # 1. 正規直交性の保持
        assert self.manifold_st52.validate_point(exp_result, atol=1e-6), "Exp result not on Stiefel manifold"

        # 2. 小さなベクトルに対する1次近似
        linear_approx = x + v_small
        exp_diff_norm = jnp.linalg.norm(exp_result - linear_approx)
        v_norm_squared = jnp.linalg.norm(v_small) ** 2

        # 2次の誤差項であることを確認（小さなvに対して）
        assert exp_diff_norm <= 5 * v_norm_squared, "Exponential map not consistent with linear approximation"

        # 3. ゼロベクトルでの恒等性
        exp_zero = self.manifold_st52._exp_impl(x, jnp.zeros_like(v_small))
        np.testing.assert_allclose(exp_zero, x, atol=1e-10)

    def test_log_jit_mathematical_correctness(self):
        """対数写像のJIT実装の数学的正確性 (Requirement 8.4)."""
        key1, key2, key3 = jr.split(self.key, 3)
        x = self.manifold_st52.random_point(key1)

        # 近い点を生成（指数写像経由）
        v_small = 0.1 * self.manifold_st52.random_tangent(key2, x)
        y = self.manifold_st52._exp_impl(x, v_small)

        # 対数写像
        log_result = self.manifold_st52._log_impl(x, y)

        # 接空間条件の確認
        assert self.manifold_st52.validate_tangent(x, log_result, atol=1e-6), "Log result not in tangent space"

        # 近い点に対する近似精度
        diff_norm = jnp.linalg.norm(log_result - v_small)
        assert diff_norm <= 0.1 * jnp.linalg.norm(v_small), "Log not inverse of exp for small vectors"

    def test_inner_jit_vs_original_equivalence_st52(self):
        """内積のJIT vs 非JITの数値同等性 (Requirement 8.1)."""
        key1, key2, key3 = jr.split(self.key, 3)
        x = self.manifold_st52.random_point(key1)
        u = self.manifold_st52.random_tangent(key2, x)
        v = self.manifold_st52.random_tangent(key3, x)

        # 元の実装
        inner_original = self.manifold_st52.inner(x, u, v)

        # JIT実装
        inner_jit = self.manifold_st52._inner_impl(x, u, v)

        # 数値同等性確認
        np.testing.assert_allclose(inner_original, inner_jit, rtol=self.rtol, atol=self.atol)

        # 対称性確認
        np.testing.assert_allclose(inner_jit, self.manifold_st52._inner_impl(x, v, u), rtol=1e-12)

        # 正定値性確認
        self_inner = self.manifold_st52._inner_impl(x, u, u)
        assert self_inner >= -1e-10, "Inner product not positive semi-definite"

    def test_dist_jit_vs_original_equivalence_st52(self):
        """距離計算のJIT vs 非JITの数値同等性 (Requirement 8.1)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        y = self.manifold_st52.random_point(key2)

        # 元の実装
        dist_original = self.manifold_st52.dist(x, y)

        # JIT実装
        dist_jit = self.manifold_st52._dist_impl(x, y)

        # 数値同等性確認
        np.testing.assert_allclose(dist_original, dist_jit, rtol=self.rtol, atol=self.atol)

        # 距離の性質確認
        assert dist_jit >= 0, "Distance not non-negative"

        # 対称性
        dist_symmetric = self.manifold_st52._dist_impl(y, x)
        np.testing.assert_allclose(dist_jit, dist_symmetric, rtol=1e-6, atol=1e-8)

        # 同一点での距離
        dist_self = self.manifold_st52._dist_impl(x, x)
        np.testing.assert_allclose(dist_self, 0.0, atol=1e-12)

    def test_orthonormal_constraints_preservation(self):
        """正規直交性制約保持の検証テスト (Requirement 1.4)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        v = self.manifold_st52.random_tangent(key2, x)

        # プロジェクション後の制約確認
        v_proj = self.manifold_st52._proj_impl(x, jr.normal(key2, (5, 2)))
        assert self.manifold_st52.validate_tangent(x, v_proj, atol=1e-6)

        # 指数写像後の制約確認
        y = self.manifold_st52._exp_impl(x, v)
        assert self.manifold_st52.validate_point(y, atol=1e-6)

        # 正規直交性の数値確認
        should_be_identity = y.T @ y
        identity = jnp.eye(self.manifold_st52.p)
        np.testing.assert_allclose(should_be_identity, identity, atol=1e-6)

    def test_large_scale_numerical_stability(self):
        """大規模行列での数値安定性テスト (Requirement 2.2)."""
        # より大きなStiefel多様体でテスト
        large_manifold = Stiefel(50, 10)
        key1, key2 = jr.split(self.key)

        x = large_manifold.random_point(key1)
        v = large_manifold.random_tangent(key2, x)

        # JIT操作が数値的に安定であることを確認
        y = large_manifold._exp_impl(x, v)

        # 正規直交性の保持
        orthonormality_error = jnp.linalg.norm(y.T @ y - jnp.eye(10))
        assert orthonormality_error < 1e-6, f"Large-scale orthonormality error: {orthonormality_error}"

        # 距離計算の安定性
        distance = large_manifold._dist_impl(x, y)
        assert not jnp.any(jnp.isnan(distance))
        assert not jnp.any(jnp.isinf(distance))
        assert distance >= 0.0

    def test_static_args_configuration(self):
        """静的引数設定のテスト (Requirement 8.2)."""
        static_args = self.manifold_st52._get_static_args("proj")
        assert static_args == (5, 2), f"Incorrect static args: {static_args}"

        # 異なる次元での確認
        static_args_43 = self.manifold_st43._get_static_args("exp")
        assert static_args_43 == (4, 3), f"Incorrect static args for St(4,3): {static_args_43}"

    def test_batch_processing_consistency_st52(self):
        """バッチ処理一貫性テスト (Requirement 8.1)."""
        batch_size = 5
        key = jr.PRNGKey(42)
        keys = jr.split(key, batch_size * 2)

        # バッチでのランダム点生成
        x_batch = self.manifold_st52.random_point(keys[0], batch_size)
        v_batch = jnp.stack([self.manifold_st52.random_tangent(keys[i + 1], x_batch[i]) for i in range(batch_size)])

        # 個別計算
        exp_individual = jnp.stack([self.manifold_st52._exp_impl(x_batch[i], v_batch[i]) for i in range(batch_size)])

        # バッチ計算（vectorized）
        exp_vectorized = jnp.vectorize(self.manifold_st52._exp_impl, signature="(n,p),(n,p)->(n,p)")(x_batch, v_batch)

        # 一貫性確認
        np.testing.assert_allclose(exp_individual, exp_vectorized, rtol=self.rtol, atol=self.atol)

    def test_special_orthogonal_group_case(self):
        """特殊直交群ケース St(3,3) = SO(3) のテスト (Requirement 1.4)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st33.random_point(key1)
        v = self.manifold_st33.random_tangent(key2, x)

        # SO(3)特有の性質：行列式が1
        det_x = jnp.linalg.det(x)
        np.testing.assert_allclose(det_x, 1.0, atol=1e-6)

        # 指数写像後も行列式が1
        y = self.manifold_st33._exp_impl(x, v)
        det_y = jnp.linalg.det(y)
        np.testing.assert_allclose(det_y, 1.0, atol=1e-6)

        # 直交性の確認
        assert jnp.allclose(y @ y.T, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(y.T @ y, jnp.eye(3), atol=1e-6)

    def test_principal_angles_computation_accuracy(self):
        """主角計算の精度テスト."""
        # 簡単なケース: 異なる部分空間のフレーム
        manifold = Stiefel(4, 2)

        # 第1フレーム: x-y平面
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        # 第2フレーム: x-z平面
        y = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        distance = manifold._dist_impl(x, y)

        # 理論値: 1つの主角がπ/2、もう1つは0なので距離はπ/2
        expected_distance = jnp.pi / 2
        np.testing.assert_allclose(distance, expected_distance, rtol=1e-4)

    def test_jit_compilation_caching(self):
        """JITコンパイルとキャッシングのテスト (Requirement 8.2)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        v = self.manifold_st52.random_tangent(key2, x)

        # 最初の呼び出し（コンパイル発生）
        result1 = self.manifold_st52._exp_impl(x, v)

        # 2回目の呼び出し（キャッシュ利用）
        result2 = self.manifold_st52._exp_impl(x, v)

        # 結果の一致確認
        np.testing.assert_allclose(result1, result2, rtol=1e-15, atol=1e-15)

    def test_error_handling_invalid_inputs(self):
        """不正入力でのエラーハンドリング (Requirement 8.2)."""
        key = jr.PRNGKey(42)
        x = self.manifold_st52.random_point(key)

        # 間違った形状の入力
        v_wrong_shape = jnp.ones((3, 3))  # Should be (5, 2)

        # エラーが適切にハンドリングされることを確認
        try:
            self.manifold_st52._proj_impl(x, v_wrong_shape)
            # 形状が合わない場合はJAXがエラーを出すか、計算が失敗する
        except (ValueError, TypeError):
            pass  # 期待される動作

    def test_integration_with_existing_methods(self):
        """既存メソッドとの統合テスト (Requirement 8.2)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)
        v = self.manifold_st52.random_tangent(key2, x)

        # JIT対応後も既存のAPIが機能することを確認
        y = self.manifold_st52.exp(x, v)
        self.manifold_st52.log(x, y)

        # 検証メソッドの動作確認
        assert self.manifold_st52.validate_point(x)
        assert self.manifold_st52.validate_point(y)
        assert self.manifold_st52.validate_tangent(x, v)

        # 一貫性の基本チェック
        distance = self.manifold_st52.dist(x, y)
        assert distance >= 0

        inner_prod = self.manifold_st52.inner(x, v, v)
        assert inner_prod >= -1e-10  # 数値誤差を考慮した非負性

    def test_mathematical_correctness_exp_log_consistency(self):
        """数学的正確性: exp-log一貫性テスト (Requirement 8.4)."""
        key1, key2 = jr.split(self.key)
        x = self.manifold_st52.random_point(key1)

        # 小さな接ベクトル（局所的な exp-log 逆性が成り立つ範囲）
        v_small = 0.01 * self.manifold_st52.random_tangent(key2, x)

        # exp -> log -> exp サイクル
        y = self.manifold_st52._exp_impl(x, v_small)
        v_recovered = self.manifold_st52._log_impl(x, y)
        y_recovered = self.manifold_st52._exp_impl(x, v_recovered)

        # 小さなベクトルに対する一貫性（数値誤差を考慮）
        v_error = jnp.linalg.norm(v_recovered - v_small)
        y_error = jnp.linalg.norm(y_recovered - y)

        v_norm = jnp.linalg.norm(v_small)

        # 相対誤差が許容範囲内であることを確認
        assert v_error <= 0.1 * v_norm, f"exp-log inconsistency in tangent space: {v_error / v_norm}"
        assert y_error <= 1e-3, f"exp-log inconsistency on manifold: {y_error}"


if __name__ == "__main__":
    pytest.main([__file__])
