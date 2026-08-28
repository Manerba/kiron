"""Declarative, immutable KIron model catalog.

Importing this package performs no I/O. Call :func:`load_catalog` explicitly to
load the packaged ``*.model.json`` resources.
"""

from .catalog import ModelCatalog
from .errors import (
    CatalogLookupError,
    CatalogResourceError,
    CatalogValidationError,
    ModelCatalogError,
)
from .models import (
    Artifact,
    ArtifactFile,
    ArtifactType,
    Backend,
    BackendType,
    Deployment,
    Loader,
    LoaderType,
    ModelEndpoint,
    ModelGroup,
    ModelTask,
    Profile,
    RequestDefault,
    Route,
    WireProfile,
)
from .parser import SCHEMA_VERSION, parse_manifest
from .resources import (
    DEFAULT_MANIFEST_PACKAGE,
    MANIFEST_SUFFIX,
    load_catalog,
    load_catalog_from_package,
)

__all__ = [
    "Artifact",
    "ArtifactFile",
    "ArtifactType",
    "Backend",
    "BackendType",
    "CatalogLookupError",
    "CatalogResourceError",
    "CatalogValidationError",
    "DEFAULT_MANIFEST_PACKAGE",
    "Deployment",
    "Loader",
    "LoaderType",
    "MANIFEST_SUFFIX",
    "ModelCatalog",
    "ModelCatalogError",
    "ModelEndpoint",
    "ModelGroup",
    "ModelTask",
    "Profile",
    "RequestDefault",
    "Route",
    "SCHEMA_VERSION",
    "WireProfile",
    "load_catalog",
    "load_catalog_from_package",
    "parse_manifest",
]
