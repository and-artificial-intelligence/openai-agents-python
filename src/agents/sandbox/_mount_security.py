from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from urllib.parse import urlsplit

from .entries.base import resolve_workspace_path
from .entries.mounts.base import (
    DockerVolumeMountStrategy,
    InContainerMountStrategy,
    Mount,
    MountStrategyBase,
)
from .entries.mounts.patterns import RcloneMountPattern
from .entries.mounts.providers.azure_blob import AzureBlobMount
from .entries.mounts.providers.box import BoxMount
from .entries.mounts.providers.gcs import GCSMount
from .entries.mounts.providers.r2 import R2Mount
from .entries.mounts.providers.s3 import S3Mount
from .entries.mounts.providers.s3_files import S3FilesMount
from .errors import InvalidManifestPathError, MountConfigError
from .workspace_paths import coerce_posix_path, posix_path_as_path

if TYPE_CHECKING:
    from .manifest import Manifest


_IN_CONTAINER_MOUNT_TYPES = frozenset(
    {
        "azure_blob_mount",
        "gcs_mount",
        "r2_mount",
        "s3_mount",
    }
)
_IN_CONTAINER_MOUNT_CLASSES = (AzureBlobMount, GCSMount, R2Mount, S3Mount)
_BUILTIN_MOUNT_TYPE_BY_CLASS: dict[type[Mount], str] = {
    mount_class: mount_class.model_fields["type"].default
    for mount_class in (
        AzureBlobMount,
        BoxMount,
        GCSMount,
        R2Mount,
        S3FilesMount,
        S3Mount,
    )
}
_BLAXEL_DRIVE_MOUNT_FIELDS = frozenset(
    {
        "description",
        "drive_mount_path",
        "drive_name",
        "drive_path",
        "drive_read_only",
        "ephemeral",
        "group",
        "is_dir",
        "mount_path",
        "mount_strategy",
        "permissions",
        "read_only",
        "type",
    }
)
_RAW_MOUNT_FIELDS_BY_TYPE: dict[str, frozenset[str]] = {
    mount_type: frozenset(mount_class.model_fields)
    for mount_class, mount_type in _BUILTIN_MOUNT_TYPE_BY_CLASS.items()
}
_RAW_MOUNT_FIELDS_BY_TYPE["blaxel_drive_mount"] = _BLAXEL_DRIVE_MOUNT_FIELDS
_BUILTIN_CLOUD_MOUNT_TYPES = frozenset(_RAW_MOUNT_FIELDS_BY_TYPE)
_SAME_PRINCIPAL_STRATEGY_TYPES = frozenset(
    {
        "blaxel_cloud_bucket",
        "daytona_cloud_bucket",
        "e2b_cloud_bucket",
        "runloop_cloud_bucket",
    }
)
_RAW_STRATEGY_FIELDS_BY_TYPE: dict[str, frozenset[str]] = {
    "blaxel_drive": frozenset({"type"}),
    "cloudflare_bucket_mount": frozenset({"type"}),
    "docker_volume": frozenset({"driver", "driver_options", "type"}),
    "in_container": frozenset({"pattern", "type"}),
    "modal_cloud_bucket": frozenset({"secret_environment_name", "secret_name", "type"}),
    "vercel_cloud_bucket": frozenset({"type"}),
}
_RAW_STRATEGY_FIELDS_BY_TYPE["docker_volume"] = frozenset(DockerVolumeMountStrategy.model_fields)
_RAW_STRATEGY_FIELDS_BY_TYPE["in_container"] = frozenset(InContainerMountStrategy.model_fields)
_RAW_RCLONE_PATTERN_FIELDS = frozenset(RcloneMountPattern.model_fields)

_CREDENTIAL_FIELDS_BY_MOUNT_TYPE: dict[str, tuple[str, ...]] = {
    "azure_blob_mount": ("account_key", "identity_client_id"),
    "box_mount": (
        "access_token",
        "box_config_file",
        "client_id",
        "client_secret",
        "config_credentials",
        "token",
    ),
    "gcs_mount": (
        "access_id",
        "access_token",
        "secret_access_key",
        "service_account_credentials",
        "service_account_file",
    ),
    "r2_mount": ("access_key_id", "secret_access_key"),
    "s3_mount": ("access_key_id", "secret_access_key", "session_token"),
}
_CREDENTIAL_FILE_FIELDS_BY_MOUNT_TYPE: dict[str, tuple[str, ...]] = {
    "box_mount": ("box_config_file",),
    "gcs_mount": ("service_account_file",),
}
_RCLONE_CONFIG_LINE_FIELDS_BY_MOUNT_TYPE: dict[str, tuple[str, ...]] = {
    "azure_blob_mount": ("account", "endpoint"),
    "gcs_mount": (),
    "r2_mount": ("account_id", "custom_domain"),
    "s3_mount": ("endpoint_url", "region", "s3_provider"),
}
_ENDPOINT_FIELDS = frozenset({"custom_domain", "endpoint", "endpoint_url"})

_HOST_MOUNT_CREDENTIAL_ENVIRONMENT_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_TENANT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "RCLONE_CONFIG",
        "RCLONE_PASSWORD_COMMAND",
    }
)


def _configured_credential_fields(mount: Mount) -> tuple[str, ...]:
    fields = _CREDENTIAL_FIELDS_BY_MOUNT_TYPE.get(mount.type, ())
    return tuple(field for field in fields if getattr(mount, field, None) is not None)


def _endpoint_contains_authority(value: str) -> bool:
    candidate = value if "://" in value or value.startswith("//") else f"//{value}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return True
    return bool(parsed.username or parsed.password or parsed.query or parsed.fragment)


def _unsafe_rclone_config_fields(values: Mapping[str, object], mount_type: str) -> tuple[str, ...]:
    unsafe: list[str] = []
    for field in _RCLONE_CONFIG_LINE_FIELDS_BY_MOUNT_TYPE.get(mount_type, ()):
        value = values.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            unsafe.append(field)
            continue
        if field == "account_id" and re.fullmatch(r"[A-Za-z0-9-]+", value) is None:
            unsafe.append(field)
            continue
        if field in _ENDPOINT_FIELDS and _endpoint_contains_authority(value):
            unsafe.append(field)
    return tuple(unsafe)


def _unsafe_typed_rclone_config_fields(mount: Mount) -> tuple[str, ...]:
    fields = _RCLONE_CONFIG_LINE_FIELDS_BY_MOUNT_TYPE.get(mount.type, ())
    return _unsafe_rclone_config_fields(
        {field: getattr(mount, field, None) for field in fields},
        mount.type,
    )


def validate_mount_credential_boundary(
    mount: Mount,
    *,
    provider_backend_id: str | None = None,
) -> None:
    """Reject mount strategies that would make cloud authority sandbox-visible."""

    strategy = mount.mount_strategy
    if type(strategy) is InContainerMountStrategy:
        if strategy.type != InContainerMountStrategy.model_fields["type"].default:
            raise MountConfigError(message="in-container mount strategy is not supported")
        expected_mount_type = _BUILTIN_MOUNT_TYPE_BY_CLASS.get(type(mount))
        if expected_mount_type is None or mount.type != expected_mount_type:
            raise MountConfigError(
                message="mount type does not support credentialless in-container mounting"
            )

        credential_fields = _configured_credential_fields(mount)
        if credential_fields:
            raise MountConfigError(
                message="in-container mounts cannot receive cloud credentials",
                context={"credential_fields": credential_fields},
            )
        if type(mount) not in _IN_CONTAINER_MOUNT_CLASSES:
            raise MountConfigError(
                message="mount type does not support credentialless in-container mounting"
            )

    strategy_backend_id = strategy._provider_backend_id
    if strategy_backend_id is not None:
        if provider_backend_id is not None and strategy_backend_id != provider_backend_id:
            raise MountConfigError(
                message="mount strategy is not supported by this sandbox backend"
            )
        return

    if type(strategy) is not InContainerMountStrategy:
        raise MountConfigError(message="mount strategy is not supported inside the sandbox")

    unsafe_config_fields = _unsafe_typed_rclone_config_fields(mount)
    if unsafe_config_fields:
        raise MountConfigError(
            message="in-container mount configuration is not safely anonymous",
            context={"credential_fields": unsafe_config_fields},
        )

    pattern = strategy.pattern
    if type(pattern) is not RcloneMountPattern:
        raise MountConfigError(message="in-container mount pattern is not supported")
    if pattern.type != RcloneMountPattern.model_fields["type"].default:
        raise MountConfigError(message="in-container mount pattern is not supported")
    if pattern.config_file_path is not None:
        raise MountConfigError(message="in-container mounts cannot use an rclone config file")
    if pattern.remote_name is not None:
        raise MountConfigError(
            message="in-container mounts cannot use an explicit rclone remote name"
        )
    if pattern.extra_args:
        raise MountConfigError(message="in-container mounts cannot use arbitrary rclone arguments")


def validate_manifest_mount_credential_boundaries(
    manifest: Manifest,
    *,
    provider_backend_id: str | None = None,
) -> None:
    root = posix_path_as_path(coerce_posix_path(manifest.root))
    entry_paths = {
        resolve_workspace_path(root, rel_path) for rel_path, _entry in manifest.iter_entries()
    }
    for mount, _mount_path in manifest.mount_targets():
        for field in _CREDENTIAL_FILE_FIELDS_BY_MOUNT_TYPE.get(mount.type, ()):
            value = getattr(mount, field, None)
            if not isinstance(value, str):
                continue
            try:
                credential_path = manifest._normalize_in_workspace_path(root, Path(value))
            except InvalidManifestPathError:
                continue
            if credential_path is not None and any(
                credential_path == entry_path or entry_path in credential_path.parents
                for entry_path in entry_paths
            ):
                raise MountConfigError(
                    message="mount credential files cannot be materialized in the sandbox",
                    context={"credential_fields": (field,)},
                )
        validate_mount_credential_boundary(mount, provider_backend_id=provider_backend_id)


def _raise_mount_state_error(
    *,
    credential_fields: tuple[str, ...],
) -> NoReturn:
    raise MountConfigError(
        message="credential-bearing mount configuration cannot be serialized or resumed",
        context={"credential_fields": credential_fields},
    )


def manifest_mount_state_is_serializable(manifest: Manifest) -> bool:
    try:
        validate_manifest_mount_credential_boundaries(manifest)
        assert_manifest_mount_state_serializable(manifest)
    except MountConfigError:
        return False
    return True


def assert_manifest_mount_state_serializable(manifest: Manifest) -> None:
    assert_raw_manifest_mounts_serializable(manifest.model_dump(mode="json"))


def inherited_mount_credential_environment_names(names: Iterable[str]) -> tuple[str, ...]:
    """Return host environment names that can grant authority to mount helpers."""

    return tuple(
        name
        for name in names
        if name in _HOST_MOUNT_CREDENTIAL_ENVIRONMENT_NAMES or name.startswith("RCLONE_")
    )


def _raw_manifest_mounts(entries: object) -> Iterator[Mapping[str, object]]:
    if not isinstance(entries, Mapping):
        return
    for entry in entries.values():
        if not isinstance(entry, Mapping):
            continue
        if "mount_strategy" in entry or entry.get("type") in _BUILTIN_CLOUD_MOUNT_TYPES:
            yield entry
        yield from _raw_manifest_mounts(entry.get("children"))


def assert_raw_manifest_mounts_serializable(manifest: Mapping[str, object]) -> None:
    """Reject credential authority in an unparsed sandbox manifest."""

    for mount in _raw_manifest_mounts(manifest.get("entries")):
        mount_type = mount.get("type")
        if not isinstance(mount_type, str) or mount_type not in _BUILTIN_CLOUD_MOUNT_TYPES:
            raise MountConfigError(message="unknown sandbox mount type cannot be persisted")
        if any(
            not isinstance(field, str) or field not in _RAW_MOUNT_FIELDS_BY_TYPE[mount_type]
            for field in mount
        ):
            raise MountConfigError(message="opaque sandbox mount fields cannot be persisted")

        configured_fields = tuple(
            field
            for field in _CREDENTIAL_FIELDS_BY_MOUNT_TYPE.get(mount_type, ())
            if mount.get(field) is not None
        )
        if configured_fields:
            _raise_mount_state_error(credential_fields=configured_fields)

        unsafe_config_fields = _unsafe_rclone_config_fields(mount, mount_type)
        if unsafe_config_fields:
            _raise_mount_state_error(credential_fields=unsafe_config_fields)

        strategy = mount.get("mount_strategy")
        if not isinstance(strategy, Mapping):
            raise MountConfigError(message="sandbox mount strategy is missing or malformed")
        strategy_type = strategy.get("type")
        if not isinstance(strategy_type, str):
            raise MountConfigError(message="sandbox mount strategy is missing or malformed")
        allowed_strategy_fields = _RAW_STRATEGY_FIELDS_BY_TYPE.get(strategy_type)
        if allowed_strategy_fields is None:
            if set(strategy) != {"type"}:
                raise MountConfigError(message="opaque sandbox mount strategy cannot be persisted")
        elif any(
            not isinstance(field, str) or field not in allowed_strategy_fields for field in strategy
        ):
            raise MountConfigError(message="opaque sandbox mount strategy cannot be persisted")

        if strategy_type == "docker_volume" and strategy.get("driver_options"):
            _raise_mount_state_error(
                credential_fields=("mount_strategy.driver_options",),
            )
        if strategy.get("secret_name") is not None:
            _raise_mount_state_error(
                credential_fields=("mount_strategy.secret_name",),
            )
        if strategy_type in _SAME_PRINCIPAL_STRATEGY_TYPES:
            raise MountConfigError(message="mount strategy is not supported inside the sandbox")
        if strategy_type != "in_container":
            continue

        pattern = strategy.get("pattern")
        pattern_type = pattern.get("type") if isinstance(pattern, Mapping) else None
        if mount_type not in _IN_CONTAINER_MOUNT_TYPES or pattern_type != "rclone":
            raise MountConfigError(message="mount does not support in-container persistence")
        assert isinstance(pattern, Mapping)
        if any(
            not isinstance(field, str) or field not in _RAW_RCLONE_PATTERN_FIELDS
            for field in pattern
        ):
            raise MountConfigError(
                message="opaque in-container mount configuration cannot be persisted"
            )
        if (
            pattern.get("config_file_path") is not None
            or pattern.get("remote_name") is not None
            or pattern.get("extra_args")
        ):
            raise MountConfigError(
                message="opaque in-container mount configuration cannot be persisted"
            )


def assert_raw_manifest_mounts_deserializable(
    manifest: Mapping[str, object],
    *,
    provider_backend_id: str,
) -> None:
    """Reject unknown or wrong-backend strategy types before state model parsing."""

    assert_raw_manifest_mounts_serializable(manifest)
    for mount in _raw_manifest_mounts(manifest.get("entries")):
        strategy = mount.get("mount_strategy")
        assert isinstance(strategy, Mapping)
        strategy_type = strategy.get("type")
        if not isinstance(strategy_type, str):
            raise MountConfigError(message="sandbox mount strategy is missing or malformed")
        strategy_class = MountStrategyBase._subclass_registry.get(strategy_type)
        if strategy_class is None:
            raise MountConfigError(message="sandbox mount strategy is unknown")
        strategy_backend_id = strategy_class._provider_backend_id
        if strategy_backend_id is not None and strategy_backend_id != provider_backend_id:
            raise MountConfigError(
                message="mount strategy is not supported by this sandbox backend"
            )


def _raw_run_state_session_payloads(
    sandbox: Mapping[str, object],
) -> Iterator[Mapping[str, object]]:
    session_state = sandbox.get("session_state")
    if isinstance(session_state, Mapping):
        yield session_state

    sessions_by_agent = sandbox.get("sessions_by_agent")
    if not isinstance(sessions_by_agent, Mapping):
        return
    for entry in sessions_by_agent.values():
        if not isinstance(entry, Mapping):
            continue
        nested_state = entry.get("session_state")
        yield nested_state if isinstance(nested_state, Mapping) else entry


def assert_run_state_sandbox_mounts_serializable(sandbox: Mapping[str, object]) -> None:
    """Reject unsafe mount authority in raw RunState payloads without provider imports."""

    for session_state in _raw_run_state_session_payloads(sandbox):
        manifest = session_state.get("manifest")
        if not isinstance(manifest, Mapping):
            continue
        assert_raw_manifest_mounts_serializable(manifest)
