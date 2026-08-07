from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from agents import Agent, RunContextWrapper
from agents.extensions.sandbox.blaxel.mounts import (
    BlaxelCloudBucketMountStrategy,
    BlaxelDriveMount,
    BlaxelDriveMountStrategy,
)
from agents.extensions.sandbox.cloudflare.mounts import CloudflareBucketMountStrategy
from agents.extensions.sandbox.daytona.mounts import DaytonaCloudBucketMountStrategy
from agents.extensions.sandbox.daytona.sandbox import DaytonaSandboxClient
from agents.extensions.sandbox.e2b.mounts import E2BCloudBucketMountStrategy
from agents.extensions.sandbox.runloop.mounts import RunloopCloudBucketMountStrategy
from agents.run_config import SandboxRunConfig
from agents.run_state import RunState
from agents.sandbox import Manifest
from agents.sandbox._mount_security import (
    assert_raw_manifest_mounts_serializable,
    manifest_mount_state_is_serializable,
    validate_mount_credential_boundary,
)
from agents.sandbox.entries import (
    AzureBlobMount,
    BoxMount,
    DockerVolumeMountStrategy,
    File,
    GCSMount,
    GitRepo,
    InContainerMountStrategy,
    LocalDir,
    MountpointMountPattern,
    R2Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3FilesMountPattern,
    S3Mount,
)
from agents.sandbox.entries.mounts.base import MountStrategyBase
from agents.sandbox.errors import MountConfigError
from agents.sandbox.runtime_session_manager import SandboxRuntimeSessionManager
from agents.sandbox.sandbox_agent import SandboxAgent
from agents.sandbox.session import (
    BaseSandboxClient,
    SandboxSession,
    SandboxSessionState,
)
from agents.sandbox.snapshot import NoopSnapshot


class _MountSecuritySessionState(SandboxSessionState):
    type: Literal["mount-security"] = "mount-security"


class _RenamedS3Mount(S3Mount):
    type: Any = "custom-mount-type-sentinel"


class _CustomExternalS3Mount(S3Mount):
    type: Any = "custom-external-mount-type-sentinel"
    api_key: str


class _CustomProviderMountStrategy(DockerVolumeMountStrategy):
    type: Any = "custom-provider-type-sentinel"
    api_key: str


class _CustomInContainerMountStrategy(InContainerMountStrategy):
    type: Any = "custom-in-container-type-sentinel"
    opaque_authority: str


class _CustomRcloneMountPattern(RcloneMountPattern):
    type: Any = "custom-rclone-type-sentinel"
    opaque_authority: str


class _MountSecurityClient(BaseSandboxClient[None]):
    backend_id = "docker"
    supports_default_options = True

    async def create(
        self,
        *,
        snapshot: object | None = None,
        manifest: Manifest | None = None,
        options: None = None,
    ) -> SandboxSession:
        _ = (snapshot, manifest, options)
        raise AssertionError("create() is not used by mount security tests")

    async def delete(self, session: SandboxSession) -> SandboxSession:
        _ = session
        raise AssertionError("delete() is not used by mount security tests")

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        _ = state
        raise AssertionError("resume() is not used by mount security tests")

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, _MountSecuritySessionState)


def _in_container_rclone() -> InContainerMountStrategy:
    return InContainerMountStrategy(pattern=RcloneMountPattern())


@pytest.mark.parametrize(
    ("mount_factory", "credential_field"),
    [
        (
            lambda: S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key="credential-sentinel",
                mount_strategy=_in_container_rclone(),
            ),
            "secret_access_key",
        ),
        (
            lambda: GCSMount(
                bucket="bucket",
                service_account_file="/workspace/service-account.json",
                mount_strategy=_in_container_rclone(),
            ),
            "service_account_file",
        ),
        (
            lambda: BoxMount(
                path="folder",
                box_config_file="/workspace/box.json",
                mount_strategy=_in_container_rclone(),
            ),
            "box_config_file",
        ),
    ],
)
def test_in_container_mount_rejects_explicit_credentials_without_echoing_values(
    mount_factory: Callable[[], object],
    credential_field: str,
) -> None:
    mount = mount_factory()
    with pytest.raises(MountConfigError) as exc_info:
        validate_mount_credential_boundary(cast("S3Mount | GCSMount | BoxMount", mount))

    error = exc_info.value
    credential_fields = cast(tuple[str, ...], error.context["credential_fields"])
    assert credential_field in credential_fields
    assert "credential-sentinel" not in str(error)
    assert "credential-sentinel" not in repr(error)


@pytest.mark.parametrize(
    "mount",
    [
        pytest.param(
            lambda: S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(config_file_path=Path("rclone.conf"))
                ),
            ),
            id="rclone-config-file",
        ),
        pytest.param(
            lambda: S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(extra_args=["--header", "secret"])
                ),
            ),
            id="rclone-extra-args",
        ),
        pytest.param(
            lambda: S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(
                        remote_name=":s3,provider=AWS,env_auth=true,credential-sentinel"
                    )
                ),
            ),
            id="rclone-explicit-remote",
        ),
        pytest.param(
            lambda: S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=MountpointMountPattern()),
            ),
            id="ambient-mountpoint",
        ),
        pytest.param(
            lambda: S3FilesMount(
                file_system_id="fs-123",
                mount_strategy=InContainerMountStrategy(pattern=S3FilesMountPattern()),
            ),
            id="ambient-s3-files",
        ),
    ],
)
def test_in_container_mount_rejects_opaque_or_ambient_authority(
    mount: Callable[[], object],
) -> None:
    configured_mount = mount()
    with pytest.raises(MountConfigError):
        validate_mount_credential_boundary(cast("S3Mount | S3FilesMount", configured_mount))


def test_provider_controlled_mount_allows_live_credentials() -> None:
    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key",
        secret_access_key="secret-key",
        mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
    )

    validate_mount_credential_boundary(mount)


def test_provider_controlled_mount_rejects_manifest_credential_file() -> None:
    manifest = Manifest(
        entries={
            "service-account.json": File(content=b"credential-file-sentinel"),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/service-account.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    )

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient._validate_create_manifest(manifest)

    assert "credential-file-sentinel" not in str(exc_info.value)
    assert "credential-file-sentinel" not in repr(exc_info.value)


def test_provider_controlled_mount_allows_external_credential_file() -> None:
    manifest = Manifest(
        entries={
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/host/service-account.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    _MountSecurityClient._validate_create_manifest(manifest)


@pytest.mark.parametrize(
    "directory_entry",
    [
        LocalDir(src=Path("host-credentials")),
        GitRepo(repo="openai/example", ref="main"),
    ],
)
def test_provider_controlled_mount_rejects_nested_materialized_credential_file(
    directory_entry: LocalDir | GitRepo,
) -> None:
    manifest = Manifest(
        entries={
            "credentials": directory_entry,
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials/service-account.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    )

    with pytest.raises(MountConfigError, match="cannot be materialized"):
        _MountSecurityClient._validate_create_manifest(manifest)


def test_custom_in_container_mount_subclass_is_rejected() -> None:
    mount = _RenamedS3Mount(
        bucket="bucket",
        secret_access_key="credential-sentinel",
        mount_strategy=_in_container_rclone(),
    )
    manifest = Manifest(entries={"data": mount})

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient._validate_create_manifest(manifest)

    assert not manifest_mount_state_is_serializable(manifest)
    assert "credential-sentinel" not in str(exc_info.value)
    assert "credential-sentinel" not in repr(exc_info.value)
    assert "custom-mount-type-sentinel" not in repr(exc_info.value)


def test_mutated_builtin_in_container_discriminators_are_rejected() -> None:
    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key",
        secret_access_key="credential-sentinel",
        mount_strategy=_in_container_rclone(),
    ).model_copy(update={"type": "mutated-mount-type-sentinel"})
    pattern = RcloneMountPattern().model_copy(update={"type": "mutated-pattern-type-sentinel"})

    with pytest.raises(MountConfigError) as mount_error:
        validate_mount_credential_boundary(mount)
    with pytest.raises(MountConfigError) as pattern_error:
        validate_mount_credential_boundary(
            S3Mount(
                bucket="bucket",
                mount_strategy=_in_container_rclone().model_copy(update={"pattern": pattern}),
            )
        )

    for error in (mount_error.value, pattern_error.value):
        assert "credential-sentinel" not in repr(error)
        assert "mutated-mount-type-sentinel" not in repr(error)
        assert "mutated-pattern-type-sentinel" not in repr(error)


@pytest.mark.parametrize(
    ("strategy", "sentinels"),
    [
        (
            _CustomInContainerMountStrategy(
                pattern=RcloneMountPattern(),
                opaque_authority="custom-strategy-authority-sentinel",
            ),
            (
                "custom-in-container-type-sentinel",
                "custom-strategy-authority-sentinel",
            ),
        ),
        (
            InContainerMountStrategy(pattern=RcloneMountPattern()).model_copy(
                update={
                    "pattern": _CustomRcloneMountPattern(
                        opaque_authority="custom-pattern-authority-sentinel"
                    )
                }
            ),
            ("custom-rclone-type-sentinel", "custom-pattern-authority-sentinel"),
        ),
    ],
)
def test_custom_in_container_strategy_shapes_are_rejected_without_echoing_values(
    strategy: MountStrategyBase,
    sentinels: tuple[str, ...],
) -> None:
    manifest = Manifest(entries={"data": S3Mount(bucket="bucket", mount_strategy=strategy)})

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient._validate_create_manifest(manifest)

    assert not manifest_mount_state_is_serializable(manifest)
    for sentinel in sentinels:
        assert sentinel not in str(exc_info.value)
        assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://user:endpoint-credential-sentinel@storage.example",
        "https://storage.example\nsecret_access_key = endpoint-credential-sentinel",
        "https://storage.example?sig=endpoint-credential-sentinel",
    ],
)
def test_in_container_mount_rejects_credential_bearing_endpoint_values(
    endpoint_url: str,
) -> None:
    mount = S3Mount(
        bucket="bucket",
        endpoint_url=endpoint_url,
        mount_strategy=_in_container_rclone(),
    )
    manifest = Manifest(entries={"data": mount})
    state = _MountSecuritySessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as create_error:
        _MountSecurityClient._validate_create_manifest(manifest)
    with pytest.raises(MountConfigError) as serialize_error:
        _MountSecurityClient().serialize_session_state(state)

    assert not manifest_mount_state_is_serializable(manifest)
    for error in (create_error.value, serialize_error.value):
        assert "endpoint-credential-sentinel" not in str(error)
        assert "endpoint-credential-sentinel" not in repr(error)


@pytest.mark.parametrize(
    "strategy",
    [
        DockerVolumeMountStrategy(driver="rclone"),
        CloudflareBucketMountStrategy(),
    ],
)
def test_provider_controlled_mount_rejects_wrong_backend(strategy: MountStrategyBase) -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key="secret-key",
                mount_strategy=strategy,
            )
        }
    )

    with pytest.raises(MountConfigError, match="not supported by this sandbox backend"):
        DaytonaSandboxClient._validate_create_manifest(manifest)


@pytest.mark.parametrize(
    "strategy",
    [
        DaytonaCloudBucketMountStrategy(),
        E2BCloudBucketMountStrategy(),
        RunloopCloudBucketMountStrategy(),
        BlaxelCloudBucketMountStrategy(),
    ],
)
def test_same_principal_hosted_mount_strategy_is_rejected(strategy: MountStrategyBase) -> None:
    mount = S3Mount(
        bucket="bucket",
        mount_strategy=strategy,
    )

    with pytest.raises(
        MountConfigError,
        match="mount strategy is not supported inside the sandbox",
    ):
        validate_mount_credential_boundary(mount)


@pytest.mark.parametrize(
    "mount",
    [
        S3Mount(bucket="bucket", mount_strategy=_in_container_rclone()),
        R2Mount(bucket="bucket", account_id="account", mount_strategy=_in_container_rclone()),
        GCSMount(bucket="bucket", mount_strategy=_in_container_rclone()),
        AzureBlobMount(
            account="account",
            container="container",
            mount_strategy=_in_container_rclone(),
        ),
    ],
)
def test_credentialless_rclone_configs_disable_ambient_auth(
    mount: S3Mount | R2Mount | GCSMount | AzureBlobMount,
) -> None:
    lines = mount._rclone_required_lines("remote")

    expected = "use_msi = false" if isinstance(mount, AzureBlobMount) else "env_auth = false"
    assert expected in lines


def test_credential_bearing_mount_state_is_not_serializable_or_resumable() -> None:
    sentinel = "credential-sentinel"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    state = _MountSecuritySessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )
    client = _MountSecurityClient()

    assert not manifest_mount_state_is_serializable(manifest)
    with pytest.raises(MountConfigError) as serialize_error:
        client.serialize_session_state(state)
    assert sentinel not in str(serialize_error.value)
    assert sentinel not in repr(serialize_error.value)

    payload = cast(dict[str, object], state.model_dump(mode="json"))
    with pytest.raises(MountConfigError) as deserialize_error:
        client.deserialize_session_state(payload)
    assert sentinel not in str(deserialize_error.value)
    assert sentinel not in repr(deserialize_error.value)


def test_custom_provider_strategy_state_is_not_serializable() -> None:
    sentinel = "custom-provider-credential-sentinel"
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    mount_strategy=_CustomProviderMountStrategy(
                        driver="rclone",
                        api_key=sentinel,
                    ),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient().serialize_session_state(state)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)
    assert "custom-provider-type-sentinel" not in repr(exc_info.value)


def test_custom_external_mount_state_is_rejected_by_direct_and_automatic_serialization() -> None:
    sentinel = "custom-external-mount-credential-sentinel"
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": _CustomExternalS3Mount(
                    bucket="bucket",
                    api_key=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    client = _MountSecurityClient()

    with pytest.raises(MountConfigError) as exc_info:
        client.serialize_session_state(state)

    agent: SandboxAgent[None] = SandboxAgent(name="agent")
    manager = SandboxRuntimeSessionManager(
        starting_agent=agent,
        sandbox_config=SandboxRunConfig(client=client),
        run_state=None,
    )
    manager.acquire_agent(agent)
    agent_id = id(agent)
    manager._current_agent_id = agent_id
    manager._resources_by_agent[agent_id] = cast(Any, SimpleNamespace(state=state))
    try:
        assert manager.serialize_resume_state() is None
    finally:
        manager._release_agents()

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)
    assert "custom-external-mount-type-sentinel" not in repr(exc_info.value)


def test_blaxel_drive_mount_state_is_raw_serializable() -> None:
    manifest = Manifest(
        entries={
            "drive": BlaxelDriveMount(
                drive_name="drive",
                mount_strategy=BlaxelDriveMountStrategy(),
            )
        }
    )

    assert manifest_mount_state_is_serializable(manifest)
    assert_raw_manifest_mounts_serializable(manifest.model_dump(mode="json"))


def test_malformed_credential_value_is_rejected_before_state_parsing() -> None:
    sentinel = "malformed-credential-sentinel"
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    payload = cast(dict[str, object], state.model_dump(mode="json"))
    manifest = cast(dict[str, object], payload["manifest"])
    entries = cast(dict[str, object], manifest["entries"])
    mount = cast(dict[str, object], entries["data"])
    mount["secret_access_key"] = {"secret": sentinel}

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient().deserialize_session_state(payload)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


def test_unknown_strategy_type_is_rejected_before_state_parsing() -> None:
    sentinel = "unknown-strategy-sentinel"
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    payload = cast(dict[str, object], state.model_dump(mode="json"))
    manifest = cast(dict[str, object], payload["manifest"])
    entries = cast(dict[str, object], manifest["entries"])
    mount = cast(dict[str, object], entries["data"])
    mount["mount_strategy"] = {"type": sentinel}

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient().deserialize_session_state(payload)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


def test_explicit_rclone_remote_is_rejected_before_state_parsing() -> None:
    sentinel = "explicit-remote-credential-sentinel"
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    mount_strategy=_in_container_rclone(),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    payload = cast(dict[str, object], state.model_dump(mode="json"))
    manifest = cast(dict[str, object], payload["manifest"])
    entries = cast(dict[str, object], manifest["entries"])
    mount = cast(dict[str, object], entries["data"])
    strategy = cast(dict[str, object], mount["mount_strategy"])
    pattern = cast(dict[str, object], strategy["pattern"])
    pattern["remote_name"] = sentinel

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient().deserialize_session_state(payload)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_explicit_session_state_is_rejected_before_provider_resume() -> None:
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    access_key_id="access-key",
                    secret_access_key="secret-key",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    client = _MountSecurityClient()
    agent: SandboxAgent[None] = SandboxAgent(name="agent")
    manager = SandboxRuntimeSessionManager(
        starting_agent=agent,
        sandbox_config=SandboxRunConfig(client=client, session_state=state),
        run_state=None,
    )
    manager.acquire_agent(agent)

    try:
        with pytest.raises(MountConfigError, match="cannot be serialized or resumed"):
            await manager.ensure_session(agent=agent, capabilities=[], is_resumed_state=True)
    finally:
        manager._release_agents()


def test_automatic_resume_state_omits_credential_bearing_mounts() -> None:
    state = _MountSecuritySessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    access_key_id="access-key",
                    secret_access_key="secret-key",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )
    client = _MountSecurityClient()
    agent: SandboxAgent[None] = SandboxAgent(name="agent")
    manager = SandboxRuntimeSessionManager(
        starting_agent=agent,
        sandbox_config=SandboxRunConfig(client=client),
        run_state=None,
    )
    manager.acquire_agent(agent)
    agent_id = id(agent)
    manager._current_agent_id = agent_id
    manager._resources_by_agent[agent_id] = cast(Any, SimpleNamespace(state=state))

    try:
        assert manager.serialize_resume_state() is None
    finally:
        manager._release_agents()


def test_manifest_backed_credential_file_is_not_serializable() -> None:
    sentinel = b"credential-file-sentinel"
    manifest = Manifest(
        entries={
            "service-account.json": File(content=sentinel),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/service-account.json",
                mount_strategy=_in_container_rclone(),
            ),
        }
    )
    state = _MountSecuritySessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )

    assert not manifest_mount_state_is_serializable(manifest)
    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient().serialize_session_state(state)

    assert sentinel.decode() not in str(exc_info.value)
    assert sentinel.decode() not in repr(exc_info.value)


def test_opaque_docker_driver_options_make_state_nonserializable() -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=DockerVolumeMountStrategy(
                    driver="rclone",
                    driver_options={"tlsport": "4049"},
                ),
            )
        }
    )

    assert not manifest_mount_state_is_serializable(manifest)


def test_run_state_to_json_rejects_raw_credential_bearing_sandbox_state() -> None:
    sentinel = "raw-run-state-credential"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    session_state = _MountSecuritySessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )
    agent: Agent[None] = Agent(name="agent")
    run_state = RunState(RunContextWrapper(context=None), "input", agent)
    run_state._sandbox = {
        "backend_id": "mount-security",
        "session_state": session_state.model_dump(mode="json"),
    }

    with pytest.raises(MountConfigError) as exc_info:
        run_state.to_json()

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_from_json_rejects_raw_credential_bearing_sandbox_state() -> None:
    agent: Agent[None] = Agent(name="agent")
    clean_state = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    clean_state["sandbox"] = {
        "backend_id": "unimported-provider",
        "sessions_by_agent": {
            "agent": {
                "agent_name": "agent",
                "session_state": {
                    "manifest": {
                        "entries": {
                            "data": {
                                "type": "gcs_mount",
                                "service_account_file": "/workspace/service-account.json",
                                "mount_strategy": {
                                    "type": "unimported_provider_strategy",
                                },
                            }
                        },
                    },
                },
            },
        },
    }

    with pytest.raises(MountConfigError, match="cannot be serialized or resumed"):
        await RunState.from_json(agent, clean_state)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mount_strategy", "use_sessions_by_agent"),
    [(None, False), ("malformed", True)],
)
async def test_run_state_rejects_raw_credentials_before_parsing_mount_strategy(
    mount_strategy: object,
    use_sessions_by_agent: bool,
) -> None:
    sentinel = "raw-missing-strategy-credential"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    raw_mount: dict[str, object] = {
        "type": "s3_mount",
        "bucket": "bucket",
        "secret_access_key": sentinel,
    }
    if mount_strategy is not None:
        raw_mount["mount_strategy"] = mount_strategy
    raw_session_state = {
        "manifest": {
            "entries": {"data": raw_mount},
        },
    }
    sandbox: dict[str, object] = {"backend_id": "unimported-provider"}
    if use_sessions_by_agent:
        sandbox["sessions_by_agent"] = {
            "agent": {
                "agent_name": "agent",
                "session_state": raw_session_state,
            }
        }
    else:
        sandbox["session_state"] = raw_session_state
    serialized["sandbox"] = sandbox

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_raw_discriminators_are_not_in_errors() -> None:
    sentinel = "raw-discriminator-sentinel"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    serialized["sandbox"] = {
        "backend_id": "unimported-provider",
        "session_state": {
            "manifest": {
                "entries": {
                    "data": {
                        "type": "s3_mount",
                        "bucket": "bucket",
                        "mount_strategy": {
                            "type": "in_container",
                            "pattern": {"type": sentinel},
                        },
                    }
                },
            },
        },
    }

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_rejects_unknown_mount_shapes_without_echoing_values() -> None:
    sentinel = "unknown-mount-credential-sentinel"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    serialized["sandbox"] = {
        "backend_id": "unimported-provider",
        "session_state": {
            "manifest": {
                "entries": {
                    "data": {
                        "type": "renamed-s3-mount",
                        "secret_access_key": sentinel,
                        "mount_strategy": {
                            "type": "in_container",
                            "pattern": {"type": "rclone"},
                        },
                    }
                },
            },
        },
    }

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_rejects_opaque_provider_strategy_fields() -> None:
    sentinel = "opaque-provider-credential-sentinel"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    serialized["sandbox"] = {
        "backend_id": "unimported-provider",
        "session_state": {
            "manifest": {
                "entries": {
                    "data": {
                        "type": "s3_mount",
                        "bucket": "bucket",
                        "mount_strategy": {
                            "type": "custom_provider",
                            "api_key": sentinel,
                        },
                    }
                },
            },
        },
    }

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque_location", ["mount", "pattern"])
async def test_run_state_rejects_opaque_mount_shape_fields(opaque_location: str) -> None:
    sentinel = f"opaque-{opaque_location}-field-sentinel"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    raw_mount: dict[str, object] = {
        "type": "s3_mount",
        "bucket": "bucket",
        "mount_strategy": {
            "type": "in_container",
            "pattern": {"type": "rclone"},
        },
    }
    if opaque_location == "mount":
        raw_mount["credential_blob"] = sentinel
    else:
        strategy = cast(dict[str, object], raw_mount["mount_strategy"])
        pattern = cast(dict[str, object], strategy["pattern"])
        pattern["opaque"] = sentinel
    serialized["sandbox"] = {
        "backend_id": "unimported-provider",
        "session_state": {"manifest": {"entries": {"data": raw_mount}}},
    }

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_rejects_credential_bearing_endpoint_values() -> None:
    sentinel = "raw-endpoint-credential-sentinel"
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    serialized["sandbox"] = {
        "backend_id": "unimported-provider",
        "session_state": {
            "manifest": {
                "entries": {
                    "data": {
                        "type": "s3_mount",
                        "bucket": "bucket",
                        "endpoint_url": f"https://user:{sentinel}@storage.example",
                        "mount_strategy": {"type": "in_container", "pattern": {"type": "rclone"}},
                    }
                },
            },
        },
    }

    with pytest.raises(MountConfigError) as exc_info:
        await RunState.from_json(agent, serialized)

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize(
    "account_id",
    [
        "user:r2-account-credential-sentinel@attacker",
        "account?r2-account-credential-sentinel",
    ],
)
def test_r2_account_id_rejects_endpoint_injection(account_id: str) -> None:
    manifest = Manifest(
        entries={
            "data": R2Mount(
                bucket="bucket",
                account_id=account_id,
                mount_strategy=_in_container_rclone(),
            )
        }
    )

    with pytest.raises(MountConfigError) as exc_info:
        _MountSecurityClient._validate_create_manifest(manifest)

    assert "r2-account-credential-sentinel" not in str(exc_info.value)
    assert "r2-account-credential-sentinel" not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_run_state_preserves_credentialless_state_before_provider_import() -> None:
    agent: Agent[None] = Agent(name="agent")
    serialized = RunState(RunContextWrapper(context=None), "input", agent).to_json()
    sandbox = {
        "backend_id": "unimported-provider",
        "session_state": {
            "type": "unimported-provider-state",
            "manifest": {
                "entries": {
                    "data": {
                        "type": "s3_mount",
                        "bucket": "public-bucket",
                        "mount_strategy": {"type": "unimported_provider_strategy"},
                    }
                }
            },
        },
    }
    serialized["sandbox"] = sandbox

    restored = await RunState.from_json(agent, serialized)

    assert restored.to_json()["sandbox"] == sandbox
