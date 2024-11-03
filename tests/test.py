import chex
import pytest
from .gen_data import create_sparse_equity_return_data, create_multivariate_equities_returns
from stochastic_fex import PPCA, PKPCA

SEED = 42
ST_SHAPE = (10000, 10)
HD_SP_SHAPE = (50, 10000)
q = 2

@pytest.fixture
def ppca_package():
    return PPCA(q=q, prior_sigma=1.0, seed=SEED)


@pytest.fixture
def pkpca_package():
    return PKPCA(q=q, prior_sigma=1.0, seed=SEED)

@pytest.fixture
def stationary_data():
    # Fixture to return stationary data
    return create_multivariate_equities_returns(n_samples=ST_SHAPE[0], n_assets=ST_SHAPE[1])


@pytest.fixture
def high_dim_sparse_data():
    # Fixture to return high-dimensional sparse data
    return create_sparse_equity_return_data(n_equities=HD_SP_SHAPE[0], n_bars=HD_SP_SHAPE[1])


def test_init(ppca_package, pkpca_package):
    ppca = ppca_package
    pkpca = pkpca_package
    assert ppca.prior_sigma == 1.0
    assert ppca.seed == SEED

    assert pkpca.prior_sigma == 1.0
    assert pkpca.seed == SEED


def test_fit_without_em(ppca_package, pkpca_package, stationary_data):
    ppca = ppca_package
    embedding = ppca.fit_transform(stationary_data, use_em=False)
    chex.assert_shape(embedding, (q, ST_SHAPE[1]))

    pkpca = pkpca_package
    embedding = pkpca.fit_transform(stationary_data, use_em=False)
    chex.assert_shape(embedding, (q, ST_SHAPE[1]))


def test_fit_with_em(ppca_package,pkpca_package,  stationary_data):
    ppca = ppca_package
    _, embedding = ppca.fit_transform(stationary_data, use_em=True, max_iter=20, verbose=1)
    chex.assert_shape(embedding, (q, ST_SHAPE[1]))

    pkpca = pkpca_package
    _, embedding = pkpca.fit_transform(stationary_data, use_em=True, max_iter=20, verbose=1)
    chex.assert_shape(embedding, (q, ST_SHAPE[1]))

def test_fit_em_high_dimensional(ppca_package, pkpca_package, high_dim_sparse_data):
    ppca = ppca_package
    _, embedding = ppca.fit_transform(high_dim_sparse_data, use_em=True)
    chex.assert_shape(embedding, (q, HD_SP_SHAPE[1]))  # Substitute with expected shape

    pkpca = pkpca_package
    _, embedding = pkpca.fit_transform(high_dim_sparse_data, use_em=True)
    chex.assert_shape(embedding, (q, HD_SP_SHAPE[1]))  # Substitute with expected shape

def test_high_dimensional_latent_dim(ppca_package, pkpca_package, high_dim_sparse_data):
    ppca = ppca_package
    _ = ppca.fit_transform(high_dim_sparse_data, use_em=True)
    transformed_data = ppca.transform()
    print(transformed_data.shape)
    chex.assert_shape(transformed_data, (q, HD_SP_SHAPE[1]))  # Substitute with expected shape
    chex.assert_rank(transformed_data, q)

    pkpca = pkpca_package
    _ = pkpca.fit_transform(high_dim_sparse_data, use_em=True)
    transformed_data = pkpca.transform()
    print(transformed_data.shape)
    chex.assert_shape(transformed_data, (q, HD_SP_SHAPE[1]))  # Substitute with expected shape
    chex.assert_rank(transformed_data, q)

def test_inverse_transform_add_noise(ppca_package, pkpca_package, stationary_data):
    ppca = ppca_package
    _ = ppca.fit_transform(stationary_data, use_em=True)
    transformed_data = ppca.transform()
    reconstructed_data = ppca.inverse_transform(transformed_data, is_add_noise=True)
    chex.assert_shape(reconstructed_data, ST_SHAPE)  # Substitute with expected shape
    chex.assert_rank(reconstructed_data, 2)

    pkpca = pkpca_package
    _ = pkpca.fit_transform(stationary_data, use_em=True)
    transformed_data = pkpca.transform()
    reconstructed_data = pkpca.inverse_transform(transformed_data, is_add_noise=True)
    chex.assert_shape(reconstructed_data, ST_SHAPE)  # Substitute with expected shape
    chex.assert_rank(reconstructed_data, 2)

def test_inverse_transform_add_noise_hd(ppca_package, pkpca_package, high_dim_sparse_data):
    ppca = ppca_package
    _ = ppca.fit_transform(high_dim_sparse_data, use_em=True)
    transformed_data = ppca.transform()
    reconstructed_data = ppca.inverse_transform(transformed_data, is_add_noise=True)
    chex.assert_shape(reconstructed_data, HD_SP_SHAPE)  # Substitute with expected shape
    chex.assert_rank(reconstructed_data, 2)

    pkpca = pkpca_package
    _ = pkpca.fit_transform(high_dim_sparse_data, use_em=True)
    transformed_data = pkpca.transform()
    reconstructed_data = pkpca.inverse_transform(transformed_data, is_add_noise=True)
    chex.assert_shape(reconstructed_data, HD_SP_SHAPE)  # Substitute with expected shape
    chex.assert_rank(reconstructed_data, 2)
