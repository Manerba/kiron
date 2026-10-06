"""Pure model-lifecycle declarations; no clients, services or native execution."""
import unittest

from kiron_common.local_inference import ModelLifecycleOperation
from embedding_provider import KironEmbeddingProvider
from ollama_provider import OllamaProvider
from prism_provider import PrismProvider


class ProviderLifecycleTests(unittest.TestCase):
    def test_model_operations_are_explicit_immutable_and_independent_of_service_control(self):
        for provider_type, expected in ((PrismProvider, {"load", "unload"}),
                                        (OllamaProvider, {"load", "unload"}),
                                        (KironEmbeddingProvider, set())):
            with self.subTest(provider=provider_type.__name__):
                provider = provider_type.__new__(provider_type)
                for service_control in (None, object()):
                    provider.service_control = service_control
                    operations = provider.model_lifecycle_operations
                    self.assertIs(type(operations), frozenset)
                    self.assertTrue(all(isinstance(value, ModelLifecycleOperation) for value in operations))
                    self.assertEqual({value.value for value in operations}, expected)
                    with self.assertRaises(AttributeError):
                        operations.add(ModelLifecycleOperation.LOAD)
                with self.assertRaises(AttributeError):
                    provider.model_lifecycle_operations = frozenset()
