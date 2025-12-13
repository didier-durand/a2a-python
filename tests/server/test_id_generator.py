"""Tests for IDGenerator abstract base class."""
import uuid
from unittest.mock import patch

import pytest

from a2a.server.id_generator import IDGeneratorContext, IDGenerator, UUIDGenerator


class TestIDGeneratorContext:
    """Tests for IDGeneratorContext."""

    def test_context_creation_with_all_fields(self):
        """Test creating context with all fields populated."""
        context = IDGeneratorContext(task_id="task_123", context_id="context_456")
        assert context.task_id == "task_123"
        assert context.context_id == "context_456"

    def test_context_creation_with_defaults(self):
        """Test creating context with default None values."""
        context = IDGeneratorContext()
        assert context.task_id is None
        assert context.context_id is None

    def test_context_creation_with_partial_fields(self):
        """Test creating context with only some fields populated."""
        context = IDGeneratorContext(task_id="task_123")
        assert context.task_id == "task_123"
        assert context.context_id is None

    def test_context_mutability(self):
        """Test that context fields can be updated (Pydantic models are mutable by default)."""
        context = IDGeneratorContext(task_id="task_123")
        context.task_id = "task_456"
        assert context.task_id == "task_456"

    def test_context_validation(self):
        """Test that context raises validation error for invalid types."""
        from pydantic import ValidationError  # pylint: disable=C0415

        with pytest.raises(ValidationError):
            IDGeneratorContext(task_id={"not": "a string"})  # noqa


class TestIDGenerator:
    """Tests for IDGenerator abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that IDGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IDGenerator()  # noqa pylint: disable=E0110

    def test_subclass_must_implement_generate(self):
        """Test that subclasses must implement the generate method."""

        class IncompleteGenerator(IDGenerator):  # noqa pylint: disable=C0115,R0903
            pass

        with pytest.raises(TypeError):
            IncompleteGenerator()  # noqa pylint: disable=E0110

    def test_valid_subclass_implementation(self):
        """Test that a valid subclass can be instantiated."""

        class ValidGenerator(IDGenerator):  # pylint: disable=C0115,R0903
            def generate(self, context: IDGeneratorContext) -> str:
                return "test_id"

        generator = ValidGenerator()
        assert generator.generate(IDGeneratorContext()) == "test_id"


class TestUUIDGenerator:
    """Tests for UUIDGenerator implementation."""

    def test_generate_returns_string(self):
        """Test that generate returns a string."""
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        result = generator.generate(context)
        assert isinstance(result, str)

    def test_generate_returns_valid_uuid(self):
        """Test that generate returns a valid UUID format."""
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        result = generator.generate(context)
        uuid.UUID(result)

    def test_generate_returns_uuid_version_4(self):
        """Test that generate returns a UUID version 4."""
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        result = generator.generate(context)
        parsed_uuid = uuid.UUID(result)
        assert parsed_uuid.version == 4

    def test_generate_ignores_context(self):
        """Test that generate ignores the context parameter."""
        generator = UUIDGenerator()
        context1 = IDGeneratorContext(task_id="task_1", context_id="context_1")
        context2 = IDGeneratorContext(task_id="task_2", context_id="context_2")
        result1 = generator.generate(context1)
        result2 = generator.generate(context2)
        uuid.UUID(result1)
        uuid.UUID(result2)
        assert result1 != result2

    def test_generate_produces_unique_ids(self):
        """Test that multiple calls produce unique IDs."""
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        ids = [generator.generate(context) for _ in range(100)]
        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_generate_with_empty_context(self):
        """Test that generate works with an empty context."""
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        result = generator.generate(context)
        assert isinstance(result, str)
        uuid.UUID(result)

    def test_generate_with_populated_context(self):
        """Test that generate works with a populated context."""
        generator = UUIDGenerator()
        context = IDGeneratorContext(task_id="task_123", context_id="context_456")
        result = generator.generate(context)
        assert isinstance(result, str)
        uuid.UUID(result)

    @patch('uuid.uuid4')
    def test_generate_calls_uuid4(self, mock_uuid4):
        """Test that generate uses uuid.uuid4() internally."""
        mock_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
        mock_uuid4.return_value = mock_uuid
        generator = UUIDGenerator()
        context = IDGeneratorContext()
        result = generator.generate(context)
        mock_uuid4.assert_called_once()
        assert result == str(mock_uuid)

    def test_generator_is_instance_of_id_generator(self):
        """Test that UUIDGenerator is an instance of IDGenerator."""
        generator = UUIDGenerator()
        assert isinstance(generator, IDGenerator)

    def test_multiple_generators_produce_different_ids(self):
        """Test that multiple generator instances produce different IDs."""
        generator1 = UUIDGenerator()
        generator2 = UUIDGenerator()
        context = IDGeneratorContext()
        result1 = generator1.generate(context)
        result2 = generator2.generate(context)
        assert result1 != result2
