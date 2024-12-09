# Probabilistic Feature Extraction in JAX

## Overview

This repository accompanies our manuscript titled **"Generative Modeling for High-Dimensional Sparse Data: Probabilistic Feature Extraction in High-Risk Financial Regimes"**. The research focuses on leveraging probabilistic generative models—specifically Probabilistic PCA (PPCA) and Probabilistic Kernel PCA (PKPCA)—to reconstruct missing financial data and explore hidden, informative patterns. Our models offer robust handling of high-dimensional, sparse financial datasets and outperform conventional methods, especially in high-volatility regimes.

The research demonstrates the superiority of PKPCA over PPCA in capturing non-linear, time-dependent features, particularly during volatile financial regimes.

1. Using information-driven bar techniques to synchronize and sample imbalanced sequence volumes.
2. Applying a sampling event-based technique, the CUMSUM Filtering method, to create strategic trading plans based on volatility.

This repository implements probabilistic dimensionality reduction models, specifically **Probabilistic PCA (PPCA)** and **Probabilistic Kernel PCA (PKPCA)**, to address two key challenges:

- Reconstructing missing values in high-dimensional time-series data.
- Extracting latent features in a sparse, information-driven bars dataset, which is a specialized form of financial sampling.

The implementation is fully **compatible with scikit-learn**, making it easy to integrate into existing machine-learning workflows.

## Key Contributions

- Implementation of **PPCA** and **PKPCA** models for dimensionality reduction and missing data imputation.
- Application of **information-driven bars** and **CUMSUM filtering** to expand asset vectors to a high-dimensional sparse multivariate setting.
- Comparison of model performance using **MSE**, **MAE**, and other metrics across different market regimes.
- Experimental validation using **risk metrics** such as **Conditional Value at Risk (CVaR)** and **Conditional Drawdown at Risk (CDaR)**.
- Seamless integration with the **scikit-learn** API for easy use in machine learning pipelines.

## 📁 Directory Structure

```bash
.
├── LICENSE
├── README.md
├── pyproject.toml
├── gen_fex
│   ├── __init__.py
│   ├── _ppcax.py
│   ├── _pkpcax.py
└── tests
    ├── __init__.py
    ├── gen_data.py
    └── test.py
```

## 🛠️ Installation and Setup Instructions

### Prerequisites

- **Python**: Ensure you have Python **3.10** or newer installed on your system.

### Installation Steps

1. **Clone the Repository**

   ```shell
   git clone https://github.com/AI-Ahmed/gen_fex.git
   cd gen_fex
   ```

2. **Install Flit**

   If you don't already have Flit installed, install it using `pip`:

   ```shell
   pip install flit
   ```

3. **Install the Package and Dependencies**

   Install the package along with its dependencies using Flit:

   ```shell
   flit install --deps develop
   ```

   This command installs the `gen_fex` package along with all required dependencies, including development and testing tools like `pytest` and `flake8`.

### Alternative: Install Directly from GitHub

If you prefer to install the package directly from GitHub without cloning the repository:

```shell
pip install git+https://github.com/AI-Ahmed/gen_fex
```

This command installs the latest version of `gen_fex` from the main branch.

### Importing the Package

After installation, you can import the PPCA model in your Python code:

```python
from gen_fex import PPCA, PKPCA
```

## 🧪 Running Tests

To run the unit tests and ensure everything is working correctly:

1. **Navigate to the Project Directory**

   If you haven't already, navigate to the project's root directory:

   ```shell
   cd gen_fex
   ```

2. **Run Tests Using pytest**

   ```shell
   pytest tests/test.py
   ```

## 📚 Usage Example

Here's a simple example of how to use the `PPCA` and `PKPCA` classs:

```python
import numpy as np
from gen_fex import PPCA, PKPCA

# Generate some sample data
data = np.random.rand(100, 1000)

# Create a PPCA, PKPCA models instances
ppca_model = PPCA(q=150)
pkpca_model = PKPCA(q=150)

# Fit the model to the data
ppca_model.fit(data, use_em=True)
pkpca_model.fit(data, use_em=True)

# Transform the data to the lower-dimensional space
transformed_data_ppca = ppca_model.transform()
transformed_data_pkpca = pkpca_model.transform()

print("PPCA Transformed Data Shape:", transformed_data_ppca.shape)
print("PKPCA Transformed Data Shape:", transformed_data_pkpca.shape)

```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE), which is a permissive open-source license that grants users extensive rights to use, modify, and distribute the software. See the [LICENSE](LICENSE) file for more details.

## 📣 Cite Our Work

If you find this work useful in your research, please consider citing:

```bibtex
@article{Atwa2024,
  author    = {Atwa, Ahmed and Sedky, Ahmed},
  title     = {Generative Modeling for High-Dimensional Sparse Data: Probabilistic Feature Extraction in High-Risk Financial Regimes},
  journal   = {},
  year      = {},
  note      = {}
}
```

---

## 🔧 Development Setup

If you're planning to contribute to the project or modify the code, follow these steps to set up your development environment:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/AI-Ahmed/gen_fex.git
   cd gen_fex
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies:

   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Flit**

   ```shell
   pip install flit
   ```

4. **Install the Package in Editable Mode**

   - For development and testing, install the package with the `test` extras:

   ```shell
   flit install --deps develop --extras test --symlink
   ```

   The `--symlink` option installs the package in editable mode, so changes to the code are immediately reflected without reinstallation.

5. **Install Pre-commit Hooks (Optional)**

   If you use `pre-commit` for code formatting and linting:

   ```shell
   pip install pre-commit
   pre-commit install
   ```

6. **Run Tests**

   ```shell
   pytest tests/test.py
   ```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📬 Contact

For any questions or inquiries, please contact [Ahmed Nabil Atwa](mailto:dr.ahmedna.ai@gmail.com).

---

## 📝 Changelog

Refer to the [CHANGELOG](CHANGELOG.md) for details on updates and changes to the project.

---

## 📦 Publishing to PyPI (Maintainers Only)

To publish a new version of the package to PyPI:

1. **Update the Version Number**

   Increment the version number in `pyproject.toml`.

2. **Build the Package**

   ```shell
   flit build
   ```

3. **Publish to PyPI**

   ```shell
   flit publish
   ```

---

## 🌐 Links

- **Documentation**: [Github Package documentation](https://github.com/AI-Ahmed/gen_fex/README.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/AI-Ahmed/gen_fex/issues)
- **Source Code**: [GitHub Repository](https://github.com/AI-Ahmed/gen_fex)
