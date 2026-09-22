"""Immutable calibration and validation dataset metadata."""

from alembic import op

from modelport.persistence import datasets

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    datasets.create(op.get_bind(), checkfirst=True)


def downgrade() -> None:
    raise RuntimeError("Restore a backup rather than discarding dataset history")
