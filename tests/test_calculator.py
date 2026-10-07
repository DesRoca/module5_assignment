import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculation import Calculation
from app.calculator import Calculator
from app.calculator_memento import CalculatorMemento
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers


def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')


def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)


def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)


def test_unexpected_operation_error(calculator):
    # Test that an unexpected error during operation execution is wrapped in OperationError
    operation = Mock()
    operation.execute.side_effect = RuntimeError('operation failed')
    calculator.operation_strategy = operation

    with pytest.raises(OperationError, match='Operation failed: operation failed'):
        calculator.perform_operation(2, 3)

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []


def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1


def test_undo_redo_when_empty(calculator):
    # Test that undo and redo return False when there is nothing to undo or redo
    assert calculator.undo() is False
    assert calculator.redo() is False


def test_undo_redo_restore_history(calculator):
    # Test that undo and redo correctly restore the history state
    calculator.set_operation(OperationFactory.create_operation('add'))
    calculator.perform_operation(2, 3)
    previous_history = calculator.history.copy()

    assert calculator.undo() is True
    assert calculator.history == []
    assert calculator.redo() is True
    assert calculator.history == previous_history

# Test History Management

def test_history_max_size(calculator):
    # Test that the history respects the maximum size limit
    calculator.config.max_history_size = 1
    calculator.set_operation(OperationFactory.create_operation('add'))

    calculator.perform_operation(2, 3)
    calculator.perform_operation(4, 5)

    assert len(calculator.history) == 1
    assert calculator.history[0].operand1 == Decimal('4')
    assert calculator.history[0].operand2 == Decimal('5')


@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()


def test_save_empty_history(calculator):
    # Test that saving an empty history creates an empty CSV file with headers
    calculator.save_history()

    saved_history = pd.read_csv(calculator.config.history_file)
    assert saved_history.empty
    assert list(saved_history.columns) == [
        'operation',
        'operand1',
        'operand2',
        'result',
        'timestamp',
    ]


def test_save_history_error(calculator):
    # Test that an OSError during history saving is wrapped in OperationError
    with patch('app.calculator.pd.DataFrame.to_csv', side_effect=OSError('save failed')):
        with pytest.raises(OperationError, match='Failed to save history: save failed'):
            calculator.save_history()


def test_logging_setup_failure(calculator):
    # Test that an OSError during logging setup is raised and logged
    with patch('app.calculator.logging.basicConfig', side_effect=OSError('log failure')):
        with pytest.raises(OSError, match='log failure'):
            Calculator(config=calculator.config)

# Test Loading History

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")


def test_load_empty_history(calculator):
    # Test that loading an empty history file results in an empty history list
    pd.DataFrame(columns=[
        'operation',
        'operand1',
        'operand2',
        'result',
        'timestamp',
    ]).to_csv(calculator.config.history_file, index=False)

    calculator.load_history()

    assert calculator.history == []


def test_load_missing_history(calculator):
    # Test that loading history when the file does not exist results in an empty history list
    calculator.config.history_file.unlink(missing_ok=True)

    calculator.load_history()

    assert calculator.history == []


def test_history_load_failure_logs_warning(calculator):
    # Test that a warning is logged when history loading fails during initialization
    with patch.object(
        Calculator,
        'load_history',
        side_effect=OperationError('History unavailable')
    ), patch('app.calculator.logging.warning') as mock_warning:
        Calculator(config=calculator.config)

    mock_warning.assert_called_once_with(
        "Could not load existing history: History unavailable"
    )


def test_load_history_error(calculator):
    # Test that an OSError during history loading is wrapped in OperationError
    with patch('app.calculator.Path.exists', return_value=True), \
         patch('app.calculator.pd.read_csv', side_effect=OSError('load failed')):
        with pytest.raises(OperationError, match='Failed to load history: load failed'):
            calculator.load_history()

# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test History Display and DataFrame Conversion

def test_get_history_dataframe(calculator):
    # Test that get_history_dataframe returns a DataFrame with the correct data
    calculator.set_operation(OperationFactory.create_operation('add'))
    calculator.perform_operation(2, 3)

    with patch('app.calculator.pd.DataFrame') as mock_dataframe:
        result = calculator.get_history_dataframe()

    assert result is mock_dataframe.return_value
    mock_dataframe.assert_called_once()
    history_data = mock_dataframe.call_args.args[0]
    assert len(history_data) == 1
    assert history_data[0]['operation'] == 'Addition'
    assert history_data[0]['operand1'] == '2'
    assert history_data[0]['operand2'] == '3'
    assert history_data[0]['result'] == '5'


def test_show_history(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    calculator.perform_operation(2, 3)

    assert calculator.show_history() == ['Addition(2, 3) = 5']


def test_last_calculation(calculator):
    # Test that last_calculation returns None 
    # when history is empty and returns the last calculation 
    # when history is not empty
    assert calculator.last_calculation() is None

    calculator.set_operation(OperationFactory.create_operation('add'))
    calculator.perform_operation(2, 3)

    assert calculator.last_calculation() is calculator.history[-1]

# Test Calculation Memento Serialization

def test_calculator_memento_to_dict():
    # Test that CalculatorMemento can be serialized to a dictionary correctly
    timestamp = datetime.datetime(2026, 10, 6, 12, 0, 0)
    calculation = Calculation(
        operation='Addition',
        operand1=Decimal('2'),
        operand2=Decimal('3'),
    )
    calculation.timestamp = timestamp
    memento = CalculatorMemento(history=[calculation], timestamp=timestamp)

    assert memento.to_dict() == {
        'history': [calculation.to_dict()],
        'timestamp': timestamp.isoformat(),
    }


def test_calculator_memento_from_dict():
    # Test that CalculatorMemento can be deserialized from a dictionary correctly
    timestamp = datetime.datetime(2026, 10, 6, 12, 0, 0)
    data = {
        'history': [{
            'operation': 'Addition',
            'operand1': '2',
            'operand2': '3',
            'result': '5',
            'timestamp': timestamp.isoformat(),
        }],
        'timestamp': timestamp.isoformat(),
    }

    memento = CalculatorMemento.from_dict(data)

    assert len(memento.history) == 1
    assert memento.history[0].operation == 'Addition'
    assert memento.history[0].operand1 == Decimal('2')
    assert memento.history[0].operand2 == Decimal('3')
    assert memento.history[0].result == Decimal('5')
    assert memento.history[0].timestamp == timestamp
    assert memento.timestamp == timestamp

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")


@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")


@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")


def _run_repl_with_calculator(inputs, calculator_mock):
    with patch('app.calculator_repl.Calculator', return_value=calculator_mock), \
         patch('builtins.input', side_effect=inputs), \
         patch('builtins.print') as mock_print:
        calculator_repl()
    return mock_print


def test_repl_exit_saves_history_failure():
    calculator_mock = Mock()
    calculator_mock.save_history.side_effect = OSError('save failed')

    mock_print = _run_repl_with_calculator(['exit'], calculator_mock)

    mock_print.assert_any_call('Warning: Could not save history: save failed')
    mock_print.assert_any_call('Goodbye!')


@pytest.mark.parametrize(
    ('command', 'history', 'expected_output'),
    [
        ('history', [], 'No calculations in history'),
        ('history', ['Addition(2, 3) = 5'], '1. Addition(2, 3) = 5'),
        ('last', None, 'No calculations have been performed yet.'),
        ('last', 'Addition(2, 3) = 5', 'Last Calculation: Addition(2, 3) = 5'),
    ],
)
def test_repl_history_and_last_commands(command, history, expected_output):
    calculator_mock = Mock()
    calculator_mock.show_history.return_value = history
    calculator_mock.last_calculation.return_value = history if isinstance(history, str) else None

    mock_print = _run_repl_with_calculator([command, 'exit'], calculator_mock)

    mock_print.assert_any_call(expected_output)


def test_repl_clear_command():
    calculator_mock = Mock()

    mock_print = _run_repl_with_calculator(['clear', 'exit'], calculator_mock)

    calculator_mock.clear_history.assert_called_once_with()
    mock_print.assert_any_call('History cleared')


@pytest.mark.parametrize(
    ('command', 'method_result', 'expected_output'),
    [
        ('undo', True, 'Operation undone'),
        ('undo', False, 'Nothing to undo'),
        ('redo', True, 'Operation redone'),
        ('redo', False, 'Nothing to redo'),
    ],
)
def test_repl_undo_and_redo_commands(command, method_result, expected_output):
    calculator_mock = Mock()
    getattr(calculator_mock, command).return_value = method_result

    mock_print = _run_repl_with_calculator([command, 'exit'], calculator_mock)

    mock_print.assert_any_call(expected_output)


@pytest.mark.parametrize('command', ['save', 'load'])
def test_repl_history_persistence_commands(command):
    calculator_mock = Mock()
    success_message = (
        'History saved successfully'
        if command == 'save'
        else 'History loaded successfully'
    )

    mock_print = _run_repl_with_calculator([command, 'exit'], calculator_mock)

    mock_print.assert_any_call(success_message)


@pytest.mark.parametrize(
    ('command', 'method_name', 'error_message'),
    [
        ('save', 'save_history', 'Error saving history: save failed'),
        ('load', 'load_history', 'Error loading history: load failed'),
    ],
)
def test_repl_history_persistence_errors(command, method_name, error_message):
    calculator_mock = Mock()
    getattr(calculator_mock, method_name).side_effect = OSError('load failed' if command == 'load' else 'save failed')

    mock_print = _run_repl_with_calculator([command, 'exit'], calculator_mock)

    mock_print.assert_any_call(error_message)


@pytest.mark.parametrize('cancel_input', [['cancel', '2'], ['2', 'cancel']])
def test_repl_operation_cancellation(cancel_input):
    calculator_mock = Mock()
    inputs = ['add', *cancel_input, 'exit']

    mock_print = _run_repl_with_calculator(inputs, calculator_mock)

    calculator_mock.perform_operation.assert_not_called()
    mock_print.assert_any_call('Operation cancelled')


@pytest.mark.parametrize(
    ('exception_type', 'expected_message'),
    [
        (ValidationError, 'Error: invalid number'),
        (OperationError, 'Error: operation failed'),
        (RuntimeError, 'Unexpected error: unexpected failure'),
    ],
)
def test_repl_operation_errors(exception_type, expected_message):
    calculator_mock = Mock()
    if exception_type is RuntimeError:
        calculator_mock.perform_operation.side_effect = exception_type('unexpected failure')
    else:
        calculator_mock.perform_operation.side_effect = exception_type('invalid number' if exception_type is ValidationError else 'operation failed')

    mock_print = _run_repl_with_calculator(['add', '2', '3', 'exit'], calculator_mock)

    mock_print.assert_any_call(expected_message)


def test_repl_unknown_command():
    calculator_mock = Mock()

    mock_print = _run_repl_with_calculator(['unknown', 'exit'], calculator_mock)

    mock_print.assert_any_call(
        "Unknown command: 'unknown'. Type 'help' for available commands."
    )


@pytest.mark.parametrize(
    ('input_exception', 'expected_message'),
    [
        (KeyboardInterrupt(), '\nOperation cancelled'),
        (EOFError(), '\nInput terminated. Exiting...'),
        (RuntimeError('input failure'), 'Error: input failure'),
    ],
)
def test_repl_input_error_handling(input_exception, expected_message):
    calculator_mock = Mock()
    inputs = [input_exception, 'exit']
    if isinstance(input_exception, EOFError):
        inputs = [input_exception]

    mock_print = _run_repl_with_calculator(inputs, calculator_mock)

    mock_print.assert_any_call(expected_message)


def test_repl_fatal_initialization_error():
    with patch('app.calculator_repl.Calculator', side_effect=RuntimeError('startup failed')), \
         patch('app.calculator_repl.logging.error') as mock_logging_error, \
         patch('builtins.print') as mock_print:
        with pytest.raises(RuntimeError, match='startup failed'):
            calculator_repl()

    mock_print.assert_any_call('Fatal error: startup failed')
    mock_logging_error.assert_called_once_with(
        'Fatal error in calculator REPL: startup failed'
    )