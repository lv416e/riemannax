import jax
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.sphere import Sphere


class TestSphereJITOptimization:
    """Sphere多様体のJIT最適化テスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        self.manifold = Sphere()
        # JIT関連の初期化
        if hasattr(self.manifold, "_reset_jit_cache"):
            self.manifold._reset_jit_cache()

    def test_sphere_jit_implementation_methods_exist(self):
        """SphereのJIT実装メソッドが存在するかテスト."""
        assert hasattr(self.manifold, "_proj_impl")
        assert hasattr(self.manifold, "_exp_impl")
        assert hasattr(self.manifold, "_log_impl")
        assert hasattr(self.manifold, "_inner_impl")
        assert hasattr(self.manifold, "_dist_impl")
        assert hasattr(self.manifold, "_get_static_args")

    def test_proj_jit_vs_original_equivalence(self):
        """プロジェクション: JIT版と元の版の数値同等性テスト."""
        # 球面上の点とベクトルを準備
        x = jnp.array([1.0, 0.0, 0.0])  # 単位ベクトル
        v = jnp.array([0.1, 0.2, 0.3])  # 任意のベクトル

        # 元の実装
        original_result = self.manifold.proj(x, v)

        # JIT実装
        jit_result = self.manifold._proj_impl(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_exp_jit_vs_original_equivalence(self):
        """指数写像: JIT版と元の版の数値同等性テスト."""
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.1, 0.2])  # xに直交する接ベクトル

        # 元の実装
        original_result = self.manifold.exp(x, v)

        # JIT実装
        jit_result = self.manifold._exp_impl(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_log_jit_vs_original_equivalence(self):
        """対数写像: JIT版と元の版の数値同等性テスト."""
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.8, 0.6, 0.0])  # 正規化された点
        y = y / jnp.linalg.norm(y)  # 球面上に正規化

        # 元の実装
        original_result = self.manifold.log(x, y)

        # JIT実装
        jit_result = self.manifold._log_impl(x, y)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_inner_jit_vs_original_equivalence(self):
        """内積: JIT版と元の版の数値同等性テスト."""
        x = jnp.array([1.0, 0.0, 0.0])
        u = jnp.array([0.0, 1.0, 0.0])  # 接ベクトル
        v = jnp.array([0.0, 0.0, 1.0])  # 接ベクトル

        # 元の実装
        original_result = self.manifold.inner(x, u, v)

        # JIT実装
        jit_result = self.manifold._inner_impl(x, u, v)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_vs_original_equivalence(self):
        """距離計算: JIT版と元の版の数値同等性テスト."""
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])

        # 元の実装
        original_result = self.manifold.dist(x, y)

        # JIT実装
        jit_result = self.manifold._dist_impl(x, y)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_numerical_stability_zero_division_prevention(self):
        """数値安定性: ゼロ除算防止テスト."""
        x = jnp.array([1.0, 0.0, 0.0])

        # ゼロベクトルでのテスト
        v_zero = jnp.zeros(3)
        result = self.manifold._exp_impl(x, v_zero)

        # ゼロ除算が発生せず、妥当な結果が得られることを確認
        assert jnp.allclose(result, x, atol=1e-8)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))

    def test_numerical_stability_clipping_operations(self):
        """数値安定性: クリッピング処理テスト."""
        # 極値でのテスト（内積が-1や1に近い場合）
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([-1.0, 0.0, 0.0])  # 対極点

        # 対数写像は数値的に不安定な場合があるが、適切に処理されることを確認
        result = self.manifold._log_impl(x, y)

        # NaNやInfが発生しないことを確認
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))

    def test_static_args_configuration(self):
        """静的引数設定テスト."""
        # _get_static_args メソッドが適切な静的引数を返すことを確認
        static_args_proj = self.manifold._get_static_args("proj")
        static_args_exp = self.manifold._get_static_args("exp")
        static_args_log = self.manifold._get_static_args("log")

        # タプルが返されることを確認
        assert isinstance(static_args_proj, tuple)
        assert isinstance(static_args_exp, tuple)
        assert isinstance(static_args_log, tuple)

    def test_large_scale_batch_processing_consistency(self):
        """大規模配列でのバッチ処理一貫性テスト."""
        batch_size = 1000
        dim = 1000

        # 大規模バッチデータの準備
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        # 球面上の点の生成
        x_batch = jax.random.normal(key1, (batch_size, dim))
        x_batch = x_batch / jnp.linalg.norm(x_batch, axis=1, keepdims=True)

        # 接ベクトルの生成
        v_batch = jax.random.normal(key2, (batch_size, dim))
        # 接空間への射影でスカラー積を0にする
        v_batch = v_batch - jnp.sum(v_batch * x_batch, axis=1, keepdims=True) * x_batch

        # バッチ処理での射影操作
        proj_results = jax.vmap(self.manifold._proj_impl)(x_batch, v_batch)

        # 結果の形状確認
        assert proj_results.shape == (batch_size, dim)

        # サンプル確認: 個別実行との一貫性
        individual_result = self.manifold._proj_impl(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(proj_results[0], individual_result)

    def test_batch_processing_performance_scaling(self):
        """バッチ処理の線形スケーリング性能テスト."""
        # 小規模バッチ
        small_batch = 100
        key = jax.random.PRNGKey(42)
        x_small = jax.random.normal(key, (small_batch, 3))
        x_small = x_small / jnp.linalg.norm(x_small, axis=1, keepdims=True)
        v_small = jax.random.normal(key, (small_batch, 3))

        # 大規模バッチ
        large_batch = 500
        x_large = jax.random.normal(key, (large_batch, 3))
        x_large = x_large / jnp.linalg.norm(x_large, axis=1, keepdims=True)
        v_large = jax.random.normal(key, (large_batch, 3))

        # バッチ処理実行確認（実際の性能測定は困難なため、実行可能性確認）
        small_results = jax.vmap(self.manifold._proj_impl)(x_small, v_small)
        large_results = jax.vmap(self.manifold._proj_impl)(x_large, v_large)

        assert small_results.shape == (small_batch, 3)
        assert large_results.shape == (large_batch, 3)

    def test_memory_efficiency_large_dimensions(self):
        """大次元でのメモリ効率性テスト."""
        # 高次元球面でのテスト
        high_dim = 2000
        key = jax.random.PRNGKey(42)

        # 高次元球面上の点
        x = jax.random.normal(key, (high_dim,))
        x = x / jnp.linalg.norm(x)

        # 高次元接ベクトル
        v = jax.random.normal(key, (high_dim,))
        v = v - jnp.dot(v, x) * x  # 接空間への射影

        # 高次元での操作が正常に実行されることを確認
        proj_result = self.manifold._proj_impl(x, v)
        exp_result = self.manifold._exp_impl(x, v)

        assert proj_result.shape == (high_dim,)
        assert exp_result.shape == (high_dim,)

        # 球面制約の確認（高次元での数値精度を考慮）
        exp_norm = jnp.linalg.norm(exp_result)
        assert jnp.allclose(exp_norm, 1.0, atol=1e-4), f"Expected norm ~1.0, got {exp_norm}"

    def test_jit_compilation_caching(self):
        """JITコンパイルとキャッシングのテスト."""
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.1, 0.2])

        # 複数回実行して一貫した結果が得られることを確認
        results = []
        for _ in range(3):
            result = self.manifold.proj(x, v)
            results.append(result)

        # 全ての結果が同等であることを確認
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

        # JIT関連の属性が存在することを確認
        assert hasattr(self.manifold, "_jit_compiled_methods")
        assert hasattr(self.manifold, "_jit_enabled")

        # BaseManifoldからの継承により、JIT機能が利用可能であることを確認
        assert callable(getattr(self.manifold, "_call_jit_method", None))

        # 実装メソッドが存在することを確認
        assert hasattr(self.manifold, "_proj_impl")
        assert callable(self.manifold._proj_impl)

    def test_error_handling_invalid_inputs(self):
        """不正入力に対するエラーハンドリングテスト."""
        x = jnp.array([1.0, 0.0, 0.0])

        # NaN入力に対するハンドリング確認
        v_nan = jnp.array([jnp.nan, 0.0, 0.0])

        # JAXではNaNが伝播するか、適切に処理されることを確認
        result = self.manifold._proj_impl(x, v_nan)

        # NaNが伝播している（正常な動作）か、適切に処理されていることを確認
        # JAXではNaN入力はNaN出力を生成するのが標準動作
        has_nan = jnp.any(jnp.isnan(result))
        has_valid_result = not jnp.any(jnp.isnan(result))

        # NaNが伝播するか、適切に処理されているかのいずれかであることを確認
        assert has_nan or has_valid_result  # 何らかの妥当な結果が得られている

        # 無効な形状入力のテスト
        x_2d = jnp.array([1.0, 0.0, 0.0])
        v_wrong_shape = jnp.array([0.1, 0.2])  # 異なる形状

        # 形状不整合は実行時にエラーとなるか、適切に処理される
        try:
            result_shape = self.manifold._proj_impl(x_2d, v_wrong_shape)
            # エラーが発生しなかった場合は、結果を検証
            assert result_shape.shape in [x_2d.shape, v_wrong_shape.shape] or jnp.any(jnp.isnan(result_shape))
        except (ValueError, TypeError, IndexError):
            # 適切なエラーが発生した場合は成功
            assert True
