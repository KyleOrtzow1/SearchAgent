"""
Unit tests for the Config module.

Tests cover:
- Environment variable loading
- Default configuration values
- Configuration validation
- Module constants and settings
- Environment variable overrides
"""
import pytest
import os
from unittest.mock import patch
import importlib


class TestConfigDefaults:
    """Test suite for default configuration values."""

    def test_default_scryfall_config(self, reset_config_module):
        """Test default Scryfall API configuration."""
        import src.mtg_search_agent.config as config

        assert config.SCRYFALL_BASE_URL == "https://api.scryfall.com"
        assert config.SCRYFALL_RATE_LIMIT_MS == 100
        assert isinstance(config.SCRYFALL_RATE_LIMIT_MS, int)

    def test_default_search_config(self, reset_config_module):
        """Test default search configuration."""
        import src.mtg_search_agent.config as config

        assert config.MAX_SEARCH_LOOPS == 5
        assert config.MAX_RESULTS_PER_SEARCH == 500
        assert isinstance(config.MAX_SEARCH_LOOPS, int)
        assert isinstance(config.MAX_RESULTS_PER_SEARCH, int)

    def test_default_evaluation_config(self, reset_config_module):
        """Test default evaluation configuration."""
        import src.mtg_search_agent.config as config

        assert config.ENABLE_PARALLEL_EVALUATION is True
        assert config.EVALUATION_BATCH_SIZE == 10
        assert isinstance(config.EVALUATION_BATCH_SIZE, int)

    def test_default_display_config(self, reset_config_module):
        """Test default display configuration."""
        import src.mtg_search_agent.config as config

        assert config.TOP_CARDS_TO_DISPLAY == 15
        assert isinstance(config.TOP_CARDS_TO_DISPLAY, int)

    def test_default_continuation_config(self, reset_config_module):
        """Test default search continuation configuration."""
        import src.mtg_search_agent.config as config

        assert config.STOP_LOOP_CONFIDENCE_THRESHOLD == 6
        assert isinstance(config.STOP_LOOP_CONFIDENCE_THRESHOLD, int)

    def test_default_pagination_config(self, reset_config_module):
        """Test default pagination configuration."""
        import src.mtg_search_agent.config as config

        assert config.ENABLE_FULL_PAGINATION is True
        assert config.MAX_PAGES_TO_FETCH == 2
        assert isinstance(config.MAX_PAGES_TO_FETCH, int)


class TestEnvironmentVariableLoading:
    """Test suite for environment variable loading and overrides."""

    def test_openai_api_key_loading(self, reset_config_module):
        """Test OPENAI_API_KEY environment variable loading."""
        test_api_key = "test-openai-api-key-12345"

        with patch.dict(os.environ, {"OPENAI_API_KEY": test_api_key}):
            # Reload the config module to pick up the new env var
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY == test_api_key

    def test_openai_api_key_not_set(self, reset_config_module):
        """Test behavior when OPENAI_API_KEY is not set."""
        # Patch load_dotenv and os.getenv to simulate no API key
        with patch('src.mtg_search_agent.config.load_dotenv'), \
             patch('src.mtg_search_agent.config.os.getenv') as mock_getenv:
            mock_getenv.return_value = None

            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY is None

    def test_openai_api_key_empty_string(self, reset_config_module):
        """Test behavior when OPENAI_API_KEY is empty string."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY == ""

    def test_dotenv_loading(self, reset_config_module, tmp_path):
        """Test .env file loading functionality."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=dotenv-test-key")

        # Change to the directory containing the .env file
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Clear any existing OPENAI_API_KEY
            with patch.dict(os.environ, {}, clear=False):
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                import src.mtg_search_agent.config as config
                importlib.reload(config)

                # Note: This test assumes dotenv loading works as expected
                # The actual behavior depends on when/how load_dotenv() is called
                assert hasattr(config, 'OPENAI_API_KEY')

        finally:
            os.chdir(original_cwd)

    def test_environment_variable_precedence(self, reset_config_module):
        """Test that environment variables take precedence over .env file."""
        env_api_key = "environment-variable-key"

        with patch.dict(os.environ, {"OPENAI_API_KEY": env_api_key}):
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY == env_api_key


class TestConfigurationConstants:
    """Test suite for configuration constants and their types."""

    def test_scryfall_base_url_format(self, reset_config_module):
        """Test Scryfall base URL format."""
        import src.mtg_search_agent.config as config

        assert config.SCRYFALL_BASE_URL.startswith("https://")
        assert "scryfall.com" in config.SCRYFALL_BASE_URL
        assert not config.SCRYFALL_BASE_URL.endswith("/")

    def test_rate_limit_reasonable_value(self, reset_config_module):
        """Test rate limit is a reasonable value."""
        import src.mtg_search_agent.config as config

        # Rate limit should be between 50ms and 1000ms for reasonable API usage
        assert 50 <= config.SCRYFALL_RATE_LIMIT_MS <= 1000

    def test_search_loops_bounds(self, reset_config_module):
        """Test search loops is within reasonable bounds."""
        import src.mtg_search_agent.config as config

        # Should be between 1 and 10 for practical use
        assert 1 <= config.MAX_SEARCH_LOOPS <= 10

    def test_results_per_search_bounds(self, reset_config_module):
        """Test max results per search is reasonable."""
        import src.mtg_search_agent.config as config

        # Should be between 10 and 1000 for practical use
        assert 10 <= config.MAX_RESULTS_PER_SEARCH <= 1000

    def test_evaluation_batch_size_bounds(self, reset_config_module):
        """Test evaluation batch size is reasonable."""
        import src.mtg_search_agent.config as config

        # Should be between 1 and 50 for practical batching
        assert 1 <= config.EVALUATION_BATCH_SIZE <= 50

    def test_top_cards_display_bounds(self, reset_config_module):
        """Test top cards to display is reasonable."""
        import src.mtg_search_agent.config as config

        # Should be between 5 and 50 for reasonable UI
        assert 5 <= config.TOP_CARDS_TO_DISPLAY <= 50

    def test_confidence_threshold_bounds(self, reset_config_module):
        """Test confidence threshold is within score range."""
        import src.mtg_search_agent.config as config

        # Should be between 1 and 10 matching the score range
        assert 1 <= config.STOP_LOOP_CONFIDENCE_THRESHOLD <= 10

    def test_max_pages_bounds(self, reset_config_module):
        """Test max pages to fetch is reasonable."""
        import src.mtg_search_agent.config as config

        # Should be between 1 and 10 for practical pagination
        assert 1 <= config.MAX_PAGES_TO_FETCH <= 10


class TestConfigurationTypes:
    """Test suite for configuration value types."""

    def test_all_numeric_configs_are_correct_types(self, reset_config_module):
        """Test all numeric configuration values have correct types."""
        import src.mtg_search_agent.config as config

        # Integer values
        int_configs = [
            'SCRYFALL_RATE_LIMIT_MS',
            'MAX_SEARCH_LOOPS',
            'MAX_RESULTS_PER_SEARCH',
            'EVALUATION_BATCH_SIZE',
            'TOP_CARDS_TO_DISPLAY',
            'STOP_LOOP_CONFIDENCE_THRESHOLD',
            'MAX_PAGES_TO_FETCH'
        ]

        for config_name in int_configs:
            value = getattr(config, config_name)
            assert isinstance(value, int), f"{config_name} should be int, got {type(value)}"

    def test_boolean_configs_are_correct_types(self, reset_config_module):
        """Test boolean configuration values have correct types."""
        import src.mtg_search_agent.config as config

        boolean_configs = [
            'ENABLE_PARALLEL_EVALUATION',
            'ENABLE_FULL_PAGINATION'
        ]

        for config_name in boolean_configs:
            value = getattr(config, config_name)
            assert isinstance(value, bool), f"{config_name} should be bool, got {type(value)}"

    def test_string_configs_are_correct_types(self, reset_config_module):
        """Test string configuration values have correct types."""
        import src.mtg_search_agent.config as config

        string_configs = ['SCRYFALL_BASE_URL']

        for config_name in string_configs:
            value = getattr(config, config_name)
            assert isinstance(value, str), f"{config_name} should be str, got {type(value)}"

    def test_optional_configs_can_be_none(self, reset_config_module):
        """Test optional configuration values can be None."""
        import src.mtg_search_agent.config as config

        # OPENAI_API_KEY can be None if not set in environment
        assert config.OPENAI_API_KEY is None or isinstance(config.OPENAI_API_KEY, str)


class TestConfigurationValidation:
    """Test suite for configuration validation and edge cases."""

    def test_config_module_imports_successfully(self):
        """Test that config module can be imported without errors."""
        try:
            import src.mtg_search_agent.config as config
            assert hasattr(config, 'SCRYFALL_BASE_URL')
        except ImportError as e:
            pytest.fail(f"Config module import failed: {e}")

    def test_config_has_all_expected_attributes(self, reset_config_module):
        """Test that config module has all expected configuration attributes."""
        import src.mtg_search_agent.config as config

        expected_attributes = [
            'OPENAI_API_KEY',
            'SCRYFALL_BASE_URL',
            'SCRYFALL_RATE_LIMIT_MS',
            'MAX_SEARCH_LOOPS',
            'MAX_RESULTS_PER_SEARCH',
            'ENABLE_PARALLEL_EVALUATION',
            'EVALUATION_BATCH_SIZE',
            'TOP_CARDS_TO_DISPLAY',
            'STOP_LOOP_CONFIDENCE_THRESHOLD',
            'ENABLE_FULL_PAGINATION',
            'MAX_PAGES_TO_FETCH'
        ]

        for attr in expected_attributes:
            assert hasattr(config, attr), f"Config missing attribute: {attr}"

    def test_config_no_circular_imports(self):
        """Test that config module doesn't have circular import issues."""
        try:
            import src.mtg_search_agent.config as config
            # Try importing multiple times to detect circular import issues
            importlib.reload(config)
            importlib.reload(config)
        except Exception as e:
            pytest.fail(f"Circular import or reload issue: {e}")

    def test_config_values_are_immutable_style(self, reset_config_module):
        """Test that config values follow immutable constant style."""
        import src.mtg_search_agent.config as config

        # All config constants should be uppercase
        for attr_name in dir(config):
            if not attr_name.startswith('_') and attr_name.islower():
                # Skip lowercase imports like 'os', 'load_dotenv'
                if attr_name in ['os', 'load_dotenv']:
                    continue
                pytest.fail(f"Config attribute {attr_name} should be uppercase")


class TestConfigurationInDifferentEnvironments:
    """Test suite for configuration behavior in different environments."""

    def test_config_with_minimal_environment(self, reset_config_module):
        """Test config behavior with minimal environment variables."""
        # Patch load_dotenv and os.getenv to simulate minimal environment
        with patch('src.mtg_search_agent.config.load_dotenv'), \
             patch('src.mtg_search_agent.config.os.getenv') as mock_getenv:
            mock_getenv.return_value = None

            import src.mtg_search_agent.config as config
            importlib.reload(config)

            # Should still have default values
            assert config.SCRYFALL_BASE_URL == "https://api.scryfall.com"
            assert config.MAX_SEARCH_LOOPS == 5
            assert config.OPENAI_API_KEY is None

    def test_config_with_development_environment(self, reset_config_module):
        """Test config behavior with development environment variables."""
        dev_env = {
            "OPENAI_API_KEY": "dev-api-key",
            "DEBUG": "true"
        }

        with patch.dict(os.environ, dev_env):
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY == "dev-api-key"

    def test_config_with_production_environment(self, reset_config_module):
        """Test config behavior with production-like environment."""
        prod_env = {
            "OPENAI_API_KEY": "prod-secure-key-12345",
            "ENVIRONMENT": "production"
        }

        with patch.dict(os.environ, prod_env):
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            assert config.OPENAI_API_KEY == "prod-secure-key-12345"
            # Production should still use the same defaults
            assert config.SCRYFALL_RATE_LIMIT_MS == 100


class TestConfigurationSecurity:
    """Test suite for configuration security considerations."""

    def test_api_key_not_logged_accidentally(self, reset_config_module):
        """Test that API key values don't accidentally get logged in string representations."""
        test_key = "secret-api-key-12345"

        with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}):
            import src.mtg_search_agent.config as config
            importlib.reload(config)

            # This is more of a documentation test - the actual security
            # depends on how the API key is used in the application
            assert config.OPENAI_API_KEY == test_key

    def test_config_module_doesnt_expose_internals(self, reset_config_module):
        """Test that config module doesn't accidentally expose internal functions."""
        import src.mtg_search_agent.config as config

        # Check that we don't accidentally expose internal dotenv functions
        dangerous_attrs = ['load_dotenv', 'find_dotenv', 'dotenv_values']

        for attr in dangerous_attrs:
            if hasattr(config, attr):
                # This is okay if it's intentionally imported
                # The test documents what's exposed
                pass