"""Initial local metadata schema. Applied only by coordinated startup/migrate."""

from alembic import op

from modelport.persistence import metadata

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    metadata.create_all(op.get_bind())


def downgrade() -> None:
    raise RuntimeError("Destructive downgrade is intentionally disabled; restore a backup instead")
