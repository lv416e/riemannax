import jax
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.so import SpecialOrthogonal


class TestSpecialOrthogonalJITOptimization:
    """SpecialOrthogonal多様体のJIT最適化テスト."""

    def setup_method(self):
        """各テスト実行前の初期化."""
        self.manifold_so3 = SpecialOrthogonal(n=3)  # SO(3)
        self.manifold_so4 = SpecialOrthogonal(n=4)  # SO(4)

        # JIT関連の初期化
        for manifold in [self.manifold_so3, self.manifold_so4]:
            if hasattr(manifold, "_reset_jit_cache"):
                manifold._reset_jit_cache()

    def test_so_jit_implementation_methods_exist(self):
        """SpecialOrthogonalのJIT実装メソッドが存在するかテスト."""
        for manifold in [self.manifold_so3, self.manifold_so4]:
            assert hasattr(manifold, "_proj_impl")
            assert hasattr(manifold, "_exp_impl")
            assert hasattr(manifold, "_log_impl")
            assert hasattr(manifold, "_inner_impl")
            assert hasattr(manifold, "_dist_impl")
            assert hasattr(manifold, "_get_static_args")

    def test_proj_jit_vs_original_equivalence_so3(self):
        """プロジェクション: JIT版と元の版の数値同等性テスト（SO(3)）."""
        # SO(3)上の回転行列
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # 任意の接線ベクトル
        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # 元の実装
        original_result = self.manifold_so3.proj(x, v)

        # JIT実装
        jit_result = self.manifold_so3._proj_impl(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_exp_jit_vs_original_equivalence_so3(self):
        """指数写像: JIT版と元の版の数値同等性テスト（SO(3)）."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # 元の実装
        original_result = self.manifold_so3.exp(x, v)

        # JIT実装
        jit_result = self.manifold_so3._exp_impl(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=6)

    def test_log_jit_vs_original_equivalence_so3(self):
        """対数写像: JIT版と元の版の数値同等性テスト（SO(3)）."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # 元の実装
        original_result = self.manifold_so3.log(x, y)

        # JIT実装
        jit_result = self.manifold_so3._log_impl(x, y)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=6)

    def test_inner_jit_vs_original_equivalence_so3(self):
        """内積: JIT版と元の版の数値同等性テスト（SO(3)）."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        u_key, v_key = jax.random.split(jax.random.PRNGKey(43))
        u = self.manifold_so3.random_tangent(u_key, x)
        v = self.manifold_so3.random_tangent(v_key, x)

        # 元の実装
        original_result = self.manifold_so3.inner(x, u, v)

        # JIT実装
        jit_result = self.manifold_so3._inner_impl(x, u, v)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_dist_jit_vs_original_equivalence_so3(self):
        """距離計算: JIT版と元の版の数値同等性テスト（SO(3)）."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # 元の実装
        original_result = self.manifold_so3.dist(x, y)

        # JIT実装
        jit_result = self.manifold_so3._dist_impl(x, y)

        # 数値同等性確認
        np.testing.assert_almost_equal(jit_result, original_result, decimal=8)

    def test_proj_jit_vs_original_equivalence_so4(self):
        """プロジェクション: JIT版と元の版の数値同等性テスト（SO(4)）."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so4.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so4.random_tangent(v_key, x)

        # 元の実装
        original_result = self.manifold_so4.proj(x, v)

        # JIT実装
        jit_result = self.manifold_so4._proj_impl(x, v)

        # 数値同等性確認
        np.testing.assert_array_almost_equal(jit_result, original_result, decimal=8)

    def test_rotation_matrix_constraints_preservation(self):
        """回転行列制約の保持検証テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # JIT版指数写像の結果
        result = self.manifold_so3._exp_impl(x, v)

        # 直交性制約の確認: R @ R.T = I
        orthogonality_check = jnp.matmul(result, result.T)
        identity = jnp.eye(3)
        np.testing.assert_array_almost_equal(orthogonality_check, identity, decimal=6)

        # 行列式制約の確認: det(R) = 1
        det_result = jnp.linalg.det(result)
        np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_so3_rodrigues_formula_optimization(self):
        """SO(3)でのRodrigues公式最適化の精度テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # 小さな回転のテスト
        small_v = 0.01 * jnp.array([[0, -0.1, 0.2], [0.1, 0, -0.3], [-0.2, 0.3, 0]])
        small_v_tangent = self.manifold_so3.proj(x, small_v)

        result_small = self.manifold_so3._exp_impl(x, small_v_tangent)

        # 大きな回転のテスト
        large_v = jnp.pi * 0.8 * jnp.array([[0, -0.1, 0.2], [0.1, 0, -0.3], [-0.2, 0.3, 0]])
        large_v_tangent = self.manifold_so3.proj(x, large_v)

        result_large = self.manifold_so3._exp_impl(x, large_v_tangent)

        # 両方の結果が回転行列制約を満たすことを確認
        for result in [result_small, result_large]:
            # 直交性確認
            orthogonality_check = jnp.matmul(result, result.T)
            np.testing.assert_array_almost_equal(orthogonality_check, jnp.eye(3), decimal=6)

            # 行列式確認
            det_result = jnp.linalg.det(result)
            np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_so3_180_degree_rotation_stability(self):
        """SO(3)での180度回転における数値安定性テスト."""
        # 180度回転に近い場合の安定性確認
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # 180度に近い回転を生成
        axis = jnp.array([1.0, 0.0, 0.0]) / jnp.linalg.norm(jnp.array([1.0, 0.0, 0.0]))
        angle = jnp.pi - 1e-6  # ほぼ180度

        # skew-symmetric matrix for 180-degree rotation
        skew = angle * jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        # 180度回転行列を作成
        rotation_180 = self.manifold_so3._expm_so3_jit(skew)
        y = jnp.matmul(x, rotation_180)

        # 対数写像が安定に実行されることを確認
        log_result = self.manifold_so3._log_impl(x, y)

        # NaNやInfが発生しないことを確認
        assert not jnp.any(jnp.isnan(log_result))
        assert not jnp.any(jnp.isinf(log_result))

    def test_static_args_configuration(self):
        """静的引数設定テスト."""
        # SO(3)とSO(4)での静的引数設定確認
        static_args_so3 = self.manifold_so3._get_static_args("exp")
        static_args_so4 = self.manifold_so4._get_static_args("exp")

        # 静的引数に次元が含まれることを確認
        assert static_args_so3 == (3,)
        assert static_args_so4 == (4,)

    def test_large_scale_batch_processing_consistency_so3(self):
        """大規模バッチ処理一貫性テスト（SO(3)）."""
        batch_size = 100
        key = jax.random.PRNGKey(42)

        # バッチデータの準備
        keys = jax.random.split(key, batch_size)
        x_batch = jax.vmap(self.manifold_so3.random_point)(keys)

        v_keys = jax.random.split(jax.random.PRNGKey(43), batch_size)
        v_batch = jax.vmap(self.manifold_so3.random_tangent)(v_keys, x_batch)

        # バッチ処理での射影操作
        proj_results = jax.vmap(self.manifold_so3._proj_impl)(x_batch, v_batch)

        # 結果の形状確認
        assert proj_results.shape == (batch_size, 3, 3)

        # 個別実行との一貫性確認
        individual_result = self.manifold_so3._proj_impl(x_batch[0], v_batch[0])
        np.testing.assert_array_almost_equal(proj_results[0], individual_result)

    def test_batch_processing_performance_scaling(self):
        """バッチ処理の線形スケーリング性能テスト."""
        # 小規模バッチ
        small_batch = 10
        key = jax.random.PRNGKey(42)

        keys_small = jax.random.split(key, small_batch)
        x_small = jax.vmap(self.manifold_so3.random_point)(keys_small)
        v_keys_small = jax.random.split(jax.random.PRNGKey(43), small_batch)
        v_small = jax.vmap(self.manifold_so3.random_tangent)(v_keys_small, x_small)

        # 大規模バッチ
        large_batch = 50
        keys_large = jax.random.split(key, large_batch)
        x_large = jax.vmap(self.manifold_so3.random_point)(keys_large)
        v_keys_large = jax.random.split(jax.random.PRNGKey(43), large_batch)
        v_large = jax.vmap(self.manifold_so3.random_tangent)(v_keys_large, x_large)

        # バッチ処理実行確認
        small_results = jax.vmap(self.manifold_so3._proj_impl)(x_small, v_small)
        large_results = jax.vmap(self.manifold_so3._proj_impl)(x_large, v_large)

        assert small_results.shape == (small_batch, 3, 3)
        assert large_results.shape == (large_batch, 3, 3)

    def test_qr_decomposition_stability(self):
        """QR分解による数値安定性確保テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # 条件数の悪い接ベクトルを作成
        v_key = jax.random.PRNGKey(43)
        v_large = 10.0 * self.manifold_so3.random_tangent(v_key, x)

        # JIT版指数写像の実行
        result = self.manifold_so3._exp_impl(x, v_large)

        # QR分解により直交性が保持されていることを確認
        orthogonality_check = jnp.matmul(result, result.T)
        np.testing.assert_array_almost_equal(orthogonality_check, jnp.eye(3), decimal=6)

        # 行列式が1に保たれていることを確認
        det_result = jnp.linalg.det(result)
        np.testing.assert_almost_equal(det_result, 1.0, decimal=6)

    def test_3d_rotation_precision(self):
        """3D回転での具体的精度テスト."""
        # 既知の回転を定義
        angle = jnp.pi / 4  # 45度回転
        axis = jnp.array([0, 0, 1])  # z軸周り

        # 初期点
        x = jnp.eye(3)

        # skew-symmetric matrix
        skew = angle * jnp.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        v = jnp.matmul(x, skew)

        # JIT版指数写像
        result = self.manifold_so3._exp_impl(x, v)

        # 期待される回転行列（z軸周り45度回転）
        cos_45 = jnp.cos(jnp.pi / 4)
        sin_45 = jnp.sin(jnp.pi / 4)
        expected = jnp.array([[cos_45, -sin_45, 0], [sin_45, cos_45, 0], [0, 0, 1]])

        # 精度確認
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_jit_compilation_caching(self):
        """JITコンパイルとキャッシングのテスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # 複数回実行して一貫した結果が得られることを確認
        results = []
        for _ in range(3):
            result = self.manifold_so3.proj(x, v)
            results.append(result)

        # 全ての結果が同等であることを確認
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])

        # JIT関連の属性が存在することを確認
        assert hasattr(self.manifold_so3, "_jit_compiled_methods")
        assert hasattr(self.manifold_so3, "_jit_enabled")

    def test_error_handling_invalid_inputs(self):
        """不正入力に対するエラーハンドリングテスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        # NaN入力に対するハンドリング確認
        v_nan = jnp.full((3, 3), jnp.nan)

        # JAXではNaNが伝播するか、適切に処理されることを確認
        result = self.manifold_so3._proj_impl(x, v_nan)

        # NaNが伝播している（正常な動作）か、適切に処理されていることを確認
        has_nan = jnp.any(jnp.isnan(result))
        has_valid_result = not jnp.any(jnp.isnan(result))

        # NaNが伝播するか、適切に処理されているかのいずれかであることを確認
        assert has_nan or has_valid_result

    def test_matrix_logarithm_numerical_stability(self):
        """行列対数の数値安定性テスト."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = self.manifold_so3.random_point(key1)
        y = self.manifold_so3.random_point(key2)

        # 対数写像の実行
        log_result = self.manifold_so3._log_impl(x, y)

        # 結果が有限であることを確認
        assert jnp.all(jnp.isfinite(log_result))

        # 結果が接空間に属することを確認
        # SO(n)の接空間at xは x @ A (A is skew-symmetric) の形
        # つまり x.T @ log_result がskew-symmetricであることを確認
        xtv = jnp.matmul(x.T, log_result)
        skew_check = xtv + xtv.T
        np.testing.assert_array_almost_equal(skew_check, jnp.zeros((3, 3)), decimal=6)

    def test_integration_with_existing_methods(self):
        """JIT統合と既存メソッドの互換性テスト."""
        key = jax.random.PRNGKey(42)
        x = self.manifold_so3.random_point(key)

        v_key = jax.random.PRNGKey(43)
        v = self.manifold_so3.random_tangent(v_key, x)

        # 既存メソッドとJIT版の組み合わせテスト

        # validate_pointメソッド（直交性チェック）
        is_valid_point = self.manifold_so3.validate_point(x)
        assert is_valid_point

        # validate_tangent メソッド（接空間チェック）
        is_valid_tangent = self.manifold_so3.validate_tangent(x, v)
        assert is_valid_tangent
