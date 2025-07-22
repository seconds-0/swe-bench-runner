"""Tests for bootstrap module."""

from pathlib import Path
from unittest.mock import patch

from swebench_runner import bootstrap, exit_codes


class TestShowWelcomeMessage:
    """Test the show_welcome_message function."""

    @patch("swebench_runner.bootstrap.click.echo")
    def test_show_welcome_message(self, mock_echo):
        """Test welcome message display."""
        bootstrap.show_welcome_message()

        # Check that welcome message was shown
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Welcome to SWE-bench Runner" in str(c) for c in calls)
        assert any("evaluate code patches" in str(c) for c in calls)


class TestShowSetupWizard:
    """Test the show_setup_wizard function."""

    @patch("swebench_runner.bootstrap.click.prompt")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_setup_wizard_skip(self, mock_echo, mock_prompt):
        """Test setup wizard when user skips."""
        mock_prompt.return_value = "3"

        bootstrap.show_setup_wizard()

        mock_prompt.assert_called_once()
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Setup Wizard" in str(c) for c in calls)
        assert any("Skipping Docker installation" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.show_macos_instructions")
    @patch("swebench_runner.bootstrap.click.prompt")
    def test_setup_wizard_macos(self, mock_prompt, mock_macos):
        """Test setup wizard when user selects macOS."""
        mock_prompt.return_value = "1"

        bootstrap.show_setup_wizard()

        mock_macos.assert_called_once()

    @patch("swebench_runner.bootstrap.show_linux_instructions")
    @patch("swebench_runner.bootstrap.click.prompt")
    def test_setup_wizard_linux(self, mock_prompt, mock_linux):
        """Test setup wizard when user selects Linux."""
        mock_prompt.return_value = "2"

        bootstrap.show_setup_wizard()

        mock_linux.assert_called_once()


class TestShowMacosInstructions:
    """Test the show_macos_instructions function."""

    @patch("webbrowser.open")
    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_macos_instructions_with_browser(self, mock_echo, mock_confirm, mock_open):
        """Test macOS instructions when user wants to open browser."""
        mock_confirm.return_value = True

        bootstrap.show_macos_instructions()

        mock_open.assert_called_once_with("https://docker.com/products/docker-desktop")
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("macOS Docker Desktop Setup" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_macos_instructions_no_browser(self, mock_echo, mock_confirm):
        """Test macOS instructions when user doesn't want browser."""
        mock_confirm.return_value = False

        bootstrap.show_macos_instructions()

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("macOS Docker Desktop Setup" in str(c) for c in calls)


class TestShowLinuxInstructions:
    """Test the show_linux_instructions function."""

    @patch("swebench_runner.bootstrap.click.echo")
    def test_linux_instructions(self, mock_echo):
        """Test Linux instructions display."""
        bootstrap.show_linux_instructions()

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Ubuntu/Debian Docker Engine Setup" in str(c) for c in calls)
        assert any("apt-get install docker.io" in str(c) for c in calls)


class TestCheckAndPromptFirstRun:
    """Test the check_and_prompt_first_run function."""

    @patch("swebench_runner.bootstrap.is_first_run")
    def test_not_first_run(self, mock_is_first):
        """Test when not first run."""
        mock_is_first.return_value = False

        result = bootstrap.check_and_prompt_first_run()

        assert result is False

    @patch("swebench_runner.bootstrap.mark_first_run_complete")
    @patch("swebench_runner.bootstrap.is_first_run")
    def test_first_run_no_input_mode(self, mock_is_first, mock_mark):
        """Test first run in CI/no-input mode."""
        mock_is_first.return_value = True

        result = bootstrap.check_and_prompt_first_run(no_input=True)

        assert result is True
        mock_mark.assert_called_once()

    @patch("swebench_runner.bootstrap.mark_first_run_complete")
    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.show_welcome_message")
    @patch("swebench_runner.bootstrap.is_first_run")
    def test_first_run_interactive_yes(
        self, mock_is_first, mock_welcome, mock_confirm, mock_mark
    ):
        """Test first run when user confirms."""
        mock_is_first.return_value = True
        mock_confirm.return_value = True

        result = bootstrap.check_and_prompt_first_run(no_input=False)

        assert result is True
        mock_welcome.assert_called_once()
        mock_mark.assert_called_once()

    @patch("swebench_runner.bootstrap.sys.exit")
    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.show_welcome_message")
    @patch("swebench_runner.bootstrap.is_first_run")
    def test_first_run_interactive_no(
        self, mock_is_first, mock_welcome, mock_confirm, mock_exit
    ):
        """Test first run when user declines."""
        mock_is_first.return_value = True
        mock_confirm.return_value = False

        bootstrap.check_and_prompt_first_run(no_input=False)

        mock_welcome.assert_called_once()
        mock_exit.assert_called_once_with(exit_codes.SUCCESS)


class TestSuggestPatchesFile:
    """Test the suggest_patches_file function."""

    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.click.echo")
    @patch("swebench_runner.bootstrap.auto_detect_patches_file")
    def test_suggest_patches_file_found_and_accepted(
        self, mock_detect, mock_echo, mock_confirm
    ):
        """Test when patches file is found and user accepts."""
        test_path = Path("patches.jsonl")
        mock_detect.return_value = test_path
        mock_confirm.return_value = True

        result = bootstrap.suggest_patches_file()

        assert result == test_path
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Found patches.jsonl" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.click.confirm")
    @patch("swebench_runner.bootstrap.auto_detect_patches_file")
    def test_suggest_patches_file_found_but_declined(self, mock_detect, mock_confirm):
        """Test when patches file is found but user declines."""
        test_path = Path("patches.jsonl")
        mock_detect.return_value = test_path
        mock_confirm.return_value = False

        result = bootstrap.suggest_patches_file()

        assert result is None

    @patch("swebench_runner.bootstrap.auto_detect_patches_file")
    def test_suggest_patches_file_not_found(self, mock_detect):
        """Test when no patches file is found."""
        mock_detect.return_value = None

        result = bootstrap.suggest_patches_file()

        assert result is None


class TestShowSuccessMessage:
    """Test the show_success_message function."""

    @patch("swebench_runner.bootstrap.click.echo")
    def test_show_success_message_first_time(self, mock_echo):
        """Test success message for first evaluation."""
        bootstrap.show_success_message("test-123", is_first_success=True)

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("🎉 SUCCESS! 🎉" in str(c) for c in calls)
        assert any(
            "Congrats on your first successful evaluation" in str(c) for c in calls
        )
        assert any("test-123" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.click.echo")
    def test_show_success_message_regular(self, mock_echo):
        """Test success message for regular evaluation."""
        bootstrap.show_success_message("test-456", is_first_success=False)

        # Should call echo exactly once with the success message
        mock_echo.assert_called_once()
        call_str = str(mock_echo.call_args)
        assert "✅" in call_str
        assert "test-456" in call_str
        assert "Evaluation completed successfully" in call_str


class TestShowDockerSetupHelp:
    """Test the show_docker_setup_help function."""

    @patch("swebench_runner.bootstrap.platform.system")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_docker_setup_help_macos(self, mock_echo, mock_system):
        """Test Docker setup help on macOS."""
        mock_system.return_value = "Darwin"

        bootstrap.show_docker_setup_help()

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("macOS users" in str(c) for c in calls)
        assert any("Docker Desktop" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.platform.system")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_docker_setup_help_linux(self, mock_echo, mock_system):
        """Test Docker setup help on Linux."""
        mock_system.return_value = "Linux"

        bootstrap.show_docker_setup_help()

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Linux users" in str(c) for c in calls)
        assert any("apt-get install docker.io" in str(c) for c in calls)

    @patch("swebench_runner.bootstrap.platform.system")
    @patch("swebench_runner.bootstrap.click.echo")
    def test_docker_setup_help_other(self, mock_echo, mock_system):
        """Test Docker setup help on other systems."""
        mock_system.return_value = "Windows"

        bootstrap.show_docker_setup_help()

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("docs.docker.com/get-docker" in str(c) for c in calls)


class TestShowResourceWarning:
    """Test the show_resource_warning function."""

    @patch("swebench_runner.bootstrap.click.echo")
    def test_show_resource_warning(self, mock_echo):
        """Test resource warning display."""
        bootstrap.show_resource_warning(30.5)

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("30.5GB" in str(c) for c in calls)
        assert any("50GB for lite dataset" in str(c) for c in calls)
        assert any("120GB+ for full dataset" in str(c) for c in calls)


class TestShowMemoryWarning:
    """Test the show_memory_warning function."""

    @patch("swebench_runner.bootstrap.click.echo")
    def test_show_memory_warning(self, mock_echo):
        """Test memory warning display."""
        bootstrap.show_memory_warning(6.0)

        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("6.0GB RAM" in str(c) for c in calls)
        assert any("8GB RAM" in str(c) for c in calls)
        assert any("16GB+ RAM" in str(c) for c in calls)
