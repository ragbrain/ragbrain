"""CLI entry point for RAGBrain"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env file from current working directory before importing anything else
# This ensures environment variables are set before pydantic-settings reads them
_env_file = Path.cwd() / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

import click
from .import_cmd import import_directory
from .dump_cmd import dump_database
from .migrate_cmd import migrate_summaries
from .namespace_cmd import export_namespaces, import_namespaces, list_namespaces


@click.group()
@click.version_option(version='0.1.0', prog_name='ragbrain')
def cli():
    """RAGBrain CLI - Manage your personal knowledge base"""
    pass


# Register commands
cli.add_command(import_directory, name='import')
cli.add_command(dump_database, name='dump')
cli.add_command(migrate_summaries, name='migrate-summaries')
cli.add_command(export_namespaces, name='namespace-export')
cli.add_command(import_namespaces, name='namespace-import')
cli.add_command(list_namespaces, name='namespace-list')


if __name__ == '__main__':
    cli()
