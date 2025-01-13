import typer
from .commands import (
    run_protein_code_assignment,
    run_smiles_to_codes,
    run_query,
    run_easy_query,
    run_download,
)

import logging
logging.basicConfig(filename="concise.log", level=logging.DEBUG)

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    """
        Learning a CoNCISE language for small-molecule binding.
    
        To download demo data and configs run `concise download`
    """
    pass


app.command("proteins_to_codes")(run_protein_code_assignment)
app.command("smiles_to_codes")(run_smiles_to_codes)
app.command("query")(run_query)
app.command("easy_query")(run_easy_query)
app.command("download")(run_download)

if __name__ == "__main__":
    app()