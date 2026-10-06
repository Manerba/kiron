import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate-ollama-runtime-contract.py")
spec = importlib.util.spec_from_file_location("validate_ollama_runtime_contract", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _container():
    return {
        "Name": "/kiron-ollama",
        "Image": "sha256:target",
        "State": {"Running": True},
        "Config": {
            "Image": "ollama/ollama:0.18.0",
            "Labels": {"com.docker.compose.service": "ollama"},
            "Env": [
                "OLLAMA_HOST=0.0.0.0:11435",
                "OLLAMA_KEEP_ALIVE=24h",
                "OLLAMA_FLASH_ATTENTION=1",
                "NVIDIA_VISIBLE_DEVICES=all",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ],
        },
        "HostConfig": {
            "PortBindings": {
                "11435/tcp": [{"HostIp": "127.0.0.1", "HostPort": "11435"}],
            },
            "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "Count": -1,
                    "Capabilities": [["gpu"]],
                },
            ],
        },
        "Mounts": [
            {
                "Type": "volume",
                "Name": "botmin_ollama",
                "Destination": "/root/.ollama",
                "RW": True,
            },
        ],
    }


def _errors(container):
    return mod.validate_contract(
        [container],
        target_image_id="sha256:target",
        expected_image="ollama/ollama:0.18.0",
    )


class RuntimeContractTests(unittest.TestCase):
    def test_accepts_expected_compose_contract(self):
        self.assertEqual(_errors(_container()), [])

    def test_rejects_missing_gpu_device_request(self):
        container = _container()
        container["HostConfig"]["DeviceRequests"] = []

        errors = _errors(container)

        self.assertTrue(any("GPU DeviceRequest" in error for error in errors))

    def test_rejects_external_or_additional_backend_binding(self):
        for address in ("", "0.0.0.0", "::", "localhost"):
            for extra in (False, True):
                container = _container()
                bindings = container["HostConfig"]["PortBindings"]["11435/tcp"]
                if not extra:
                    bindings.clear()
                bindings.append({"HostIp": address, "HostPort": "11435"})
                self.assertTrue(any("PortBinding" in error for error in _errors(container)))

    def test_rejects_wrong_volume(self):
        container = _container()
        container["Mounts"][0]["Name"] = "other_volume"

        errors = _errors(container)

        self.assertTrue(any("botmin_ollama" in error for error in errors))

    def test_rejects_wrong_restart_policy(self):
        container = _container()
        container["HostConfig"]["RestartPolicy"] = {"Name": "always"}

        errors = _errors(container)

        self.assertTrue(any("RestartPolicy" in error for error in errors))

    def test_rejects_missing_env(self):
        container = _container()
        container["Config"]["Env"] = [
            item for item in container["Config"]["Env"]
            if not item.startswith("NVIDIA_VISIBLE_DEVICES=")
        ]

        errors = _errors(container)

        self.assertTrue(any("NVIDIA_VISIBLE_DEVICES" in error for error in errors))

    def test_rejects_unexpected_bound_port(self):
        container = _container()
        container["HostConfig"]["PortBindings"]["11434/tcp"] = [
            {"HostIp": "", "HostPort": "11434"}
        ]

        errors = _errors(container)

        self.assertTrue(any("Unerwartete Host-Portbindings" in error for error in errors))

    def test_rejects_non_compose_container(self):
        container = _container()
        container["Config"]["Labels"] = {}

        errors = _errors(container)

        self.assertTrue(any("Compose-Service" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
