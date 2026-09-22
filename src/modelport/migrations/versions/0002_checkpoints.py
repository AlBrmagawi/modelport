"""Complete native-step checkpoints; never exposed as usable artifacts."""

from alembic import op

from modelport.persistence import checkpoints

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    checkpoints.create(op.get_bind(), checkfirst=True)


def downgrade() -> None:
    raise RuntimeError("Restore a backup rather than discarding checkpoint history")
