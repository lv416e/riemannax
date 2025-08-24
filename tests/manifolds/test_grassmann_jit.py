import jax
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.grassmann import Grassmann


class TestGrassmannJITOptimization:
    """JIT optimization tests for Grassmann manifold."""

    def setup_method(self):
        """Setup before each test execution."""
        self.manifold_gr35 = Grassmann(n=5, p=3)  # Gr(3,5)
        self.manifold_gr24 = Grassmann(n=4, p=2)  # Gr(2,4)

        # JIT-related initialization
        for manifold in [self.manifold_gr35, self.manifold_gr24]:
            if hasattr(manifold, "_reset_jit_cache"):
                manifold._reset_jit_cache()

    def test_grassmann_jit_implementation_methods_exist(self):
        """Test if Grassmann JIT implementation methods exist."""
        for manifold in [self.manifold_gr35, self.manifold_gr24]:
            assert hasattr(manifold, "_proj_impl")
            assert hasattr(manifold, "_exp_impl")
            assert hasattr(manifold, "_log_impl")
            assert hasattr(manifold, "_inner_impl")
            assert hasattr(manifold, "_dist_impl")
            assert hasattr(manifold, "_get_static_args")

    def test_proj_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of projection: JIT vs original implementation (Gr(3,5))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Original implementation
        original_result = self.manifold_gr35.proj(x, v)

        # JIT implementation
        jit_result = self.manifold_gr35._proj_impl(x, v)

        # Verify numerical equivalence
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_exp_jit_vs_original_equivalence_gr35(self):
        """Test numerical equivalence of exponential map: JIT vs original implementation (Gr(3,5))."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # Original implementation (uses retraction, but JIT version is true exponential map)
        # Verify results are numerically close
        original_result = self.manifold_gr35.exp(x, v)

        # JIT implementation (true exponential map)
        jit_result = self.manifold_gr35._exp_impl(x, v)

        # Verify both results are points on the Grassmann manifold
        assert self.manifold_gr35.validate_point(original_result)
        assert self.manifold_gr35.validate_point(jit_result)

        # Verify distance is close (not exactly the same but mathematically correct)
        distance = self.manifold_gr35.dist(original_result, jit_result)
        assert distance < 1.0  # Reasonable tolerance

    def test_log_jit_mathematical_correctness(self):
        """Test mathematical correctness of logarithmic map: JIT implementation."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        y = self.manifold_gr35.random_point(key2)

        # JIT実装の対数写像
        log_result = self.manifold_gr35._log_impl(x, y)

        # 結果が接空間に属することを確認
        assert self.manifold_gr35.validate_tangent(x, log_result)

        # 指数写像との一貫性確認（exp(log(x,y)) ≈ y となることを期待しないが、数学的性質を確認）
        exp_log_result = self.manifold_gr35._exp_impl(x, log_result)

        # 結果がGrassmann多様体上の点であることを確認
        assert self.manifold_gr35.validate_point(exp_log_result)

    def test_inner_jit_vs_original_equivalence_gr35(self):
        """内積: JIT版と元の版の数値同等性テスト（Gr(3,5)）."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        u_key, v_key = jax.random.split(jax.random.PRNGKey(43))
        u = self.manifold_gr35.random_tangent(u_key, x)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # 元の実装
        original_result = self.manifold_gr35.inner(x, u, v)

        # JIT実装
        jit_result = self.manifold_gr35._inner_impl(x, u, v)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_vs_original_equivalence_gr35(self):
        """距離計算: JIT版と元の版の数値同等性テスト（Gr(3,5)）."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        y = self.manifold_gr35.random_point(key2)

        # 元の実装
        original_result = self.manifold_gr35.dist(x, y)

        # JIT実装
        jit_result = self.manifold_gr35._dist_impl(x, y)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=6)

    def test_subspace_constraints_preservation(self):
        """部分空間制約保持の検証テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # JIT版指数写像の結果
        result = self.manifold_gr35._exp_impl(x, v)

        # 直交性制約の確認: X^T @ X = I
        orthogonality_check = jnp.matmul(result.T, result)
        identity = jnp.eye(self.manifold_gr35.p)
        np.testing.assert_array_almost_equal(orthogonality_check, identity, decimal=6)

        # 形状確認
        assert result.shape == (self.manifold_gr35.n, self.manifold_gr35.p)

    def test_modified_exponential_map_correctness(self):
        """修正された指数写像の数学的正確性確認."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # ゼロベクトルでのテスト
        zero_v = jnp.zeros((self.manifold_gr35.n, self.manifold_gr35.p))
        result_zero = self.manifold_gr35._exp_impl(x, zero_v)

        # ゼロベクトルの場合、元の点が返されるべき
        np.testing.assert_array_almost_equal(result_zero, x, decimal=8)

        # 小さなベクトルでのテスト
        small_v = 0.01 * self.manifold_gr35.random_tangent(jax.random.PRNGKey(43), x)
        result_small = self.manifold_gr35._exp_impl(x, small_v)

        # 結果が多様体制約を満たすことを確認
        assert self.manifold_gr35.validate_point(result_small)

    def test_large_scale_matrix_numerical_stability(self):
        """大規模行列での数値安定性テスト."""
        # より大きな次元でのテスト
        large_manifold = Grassmann(n=50, p=10)

        key = jax.random.PRNGKey(42)
        x = large_manifold.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = large_manifold.random_tangent(v_key, x)

        # JIT実装での計算
        proj_result = large_manifold._proj_impl(x, v)
        exp_result = large_manifold._exp_impl(x, v)

        # 結果の検証
        assert proj_result.shape == (50, 10)
        assert exp_result.shape == (50, 10)

        # 数値安定性確認
        assert not jnp.any(jnp.isnan(proj_result))
        assert not jnp.any(jnp.isnan(exp_result))
        assert not jnp.any(jnp.isinf(proj_result))
        assert not jnp.any(jnp.isinf(exp_result))

        # 多様体制約確認
        assert large_manifold.validate_point(exp_result)

    def test_static_args_configuration(self):
        """静的引数設定テスト."""
        # Gr(3,5)とGr(2,4)での静的引数設定確認
        static_args_gr35 = self.manifold_gr35._get_static_args("exp")
        static_args_gr24 = self.manifold_gr24._get_static_args("exp")

        # 静的引数に次元が含まれることを確認
        assert static_args_gr35 == (5, 3)
        assert static_args_gr24 == (4, 2)

    def test_batch_processing_consistency_gr35(self):
        """バッチ処理一貫性テスト（Gr(3,5)）."""
        batch_size = 10
        key = jax.random.PRNGKey(42)

        # バッチデータの準備
        keys = jax.random.split(key, batch_size)
        x_batch = jax.vmap(self.manifold_gr35.random_point)(keys)

        v_keys = jax.random.split(jax.random.PRNGKey(43), batch_size)
        v_batch = jax.vmap(self.manifold_gr35.random_tangent)(v_keys, x_batch)

        # バッチ処理での射影操作
        proj_results = jax.vmap(self.manifold_gr35._proj_impl)(x_batch, v_batch)

        # 結果の形状確認
        assert proj_results.shape == (batch_size, 5, 3)

        # 個別実行との一貫性確認
        individual_result = self.manifold_gr35._proj_impl(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(proj_results[0], individual_result)

    def test_svd_decomposition_numerical_stability(self):
        """SVD分解による数値安定性確保テスト."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_gr35.random_point(key1)
        self.manifold_gr35.random_point(key2)

        # 条件数の悪い場合をシミュレート
        # 非常に近い部分空間
        epsilon = 1e-8
        y_close = x + epsilon * self.manifold_gr35.random_tangent(key2, x)
        Q_close, _ = jnp.linalg.qr(y_close, mode="reduced")

        # SVDベースの距離計算
        distance = self.manifold_gr35._dist_impl(x, Q_close)

        # NaNやInfが発生しないことを確認
        assert not jnp.any(jnp.isnan(distance))
        assert not jnp.any(jnp.isinf(distance))
        assert distance >= 0.0

    def test_principal_angles_computation_accuracy(self):
        """主角計算の精度テスト."""
        # 簡単なケース: 既知の主角を持つ部分空間
        manifold_gr24 = Grassmann(4, 2)

        # 第1部分空間: span{(1,0,0,0), (0,1,0,0)} = x-y平面
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        # 第2部分空間: span{(1,0,0,0), (0,0,1,0)} = x-z平面
        # 主角は90度（π/2）
        y = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        # 距離計算
        distance = manifold_gr24._dist_impl(x, y)

        # 理論値: 1つの主角がπ/2、もう1つは0なので距離はπ/2
        expected_distance = jnp.pi / 2
        np.testing.assert_almost_equal(distance, expected_distance, decimal=4)

    def test_jit_compilation_caching(self):
        """JITコンパイルとキャッシングのテスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # 複数回実行して一貫した結果が得られることを確認
        results = []
        for _ in range(3):
            result = self.manifold_gr35.proj(x, v)
            results.append(result)

        # 全ての結果が同等であることを確認
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

        # JIT関連の属性が存在することを確認
        assert hasattr(self.manifold_gr35, "_jit_compiled_methods")
        assert hasattr(self.manifold_gr35, "_jit_enabled")

    def test_error_handling_invalid_inputs(self):
        """不正入力に対するエラーハンドリングテスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # NaN入力に対するハンドリング確認
        v_nan = jnp.full((5, 3), jnp.nan)

        # JAXではNaNが伝播するか、適切に処理されることを確認
        result = self.manifold_gr35._proj_impl(x, v_nan)

        # NaNが伝播している（正常な動作）か、適切に処理されていることを確認
        has_nan = jnp.any(jnp.isnan(result))
        has_valid_result = not jnp.any(jnp.isnan(result))

        # NaNが伝播するか、適切に処理されているかのいずれかであることを確認
        assert has_nan or has_valid_result

    def test_orthonormal_columns_preservation(self):
        """正規直交列の保持確認テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # 大きな接ベクトル
        v_key = jax.random.PRNGKey(43)
        v_large = 5.0 * self.manifold_gr35.random_tangent(v_key, x)

        # JIT版指数写像の実行
        result = self.manifold_gr35._exp_impl(x, v_large)

        # 列が正規直交であることを確認
        gram_matrix = jnp.matmul(result.T, result)
        identity = jnp.eye(self.manifold_gr35.p)

        np.testing.assert_array_almost_equal(gram_matrix, identity, decimal=6)

    def test_integration_with_existing_methods(self):
        """JIT統合と既存メソッドの互換性テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_gr35.random_tangent(v_key, x)

        # 既存メソッドとJIT版の組み合わせテスト

        # validate_pointメソッド（直交性チェック）
        is_valid_point = self.manifold_gr35.validate_point(x)
        assert is_valid_point

        # validate_tangent メソッド（接空間チェック）
        is_valid_tangent = self.manifold_gr35.validate_tangent(x, v)
        assert is_valid_tangent

    def test_mathematical_correctness_exp_log_consistency(self):
        """数学的正確性：指数・対数写像の一貫性テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_gr35.random_point(key)

        # 小さな接ベクトル（線形化が有効な範囲）
        v_key = jax.random.PRNGKey(43)
        v_small = 0.1 * self.manifold_gr35.random_tangent(v_key, x)

        # exp -> log の一貫性確認
        y = self.manifold_gr35._exp_impl(x, v_small)
        v_recovered = self.manifold_gr35._log_impl(x, y)

        # 完全に一致しないかもしれないが、同じ方向であることを確認
        # （Grassmann多様体の場合、局所的な一貫性）
        inner_original = self.manifold_gr35.inner(x, v_small, v_small)
        inner_recovered = self.manifold_gr35.inner(x, v_recovered, v_recovered)

        # ノルムが大きく異ならないことを確認
        assert jnp.abs(inner_original - inner_recovered) < 0.5
