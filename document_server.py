from pathlib import Path
from urllib.parse import unquote
import mimetypes

from flask import Flask, request, abort, send_file
from flask import Response

###############################################################################
# Configuration
###############################################################################

HOST = "127.0.0.1"
PORT = 8000

DOCUMENTS_FOLDER = Path("./documents").resolve()

###############################################################################
# Flask App
###############################################################################

app = Flask(__name__)

###############################################################################
# Helper Functions
###############################################################################


def safe_document_path(filename: str) -> Path:
    """
    Convert the requested filename into an absolute path while preventing
    directory traversal attacks.
    """

    filename = unquote(filename)

    filepath = (DOCUMENTS_FOLDER / filename).resolve()

    if DOCUMENTS_FOLDER not in filepath.parents and filepath != DOCUMENTS_FOLDER:
        abort(403, description="Access denied.")

    if not filepath.exists():
        abort(404, description="Document not found.")

    if not filepath.is_file():
        abort(404)

    return filepath


###############################################################################
# Routes
###############################################################################

@app.route("/")
def index():
    return "Askly Document Server Running"


@app.route("/view_source")
def view_source():

    filename = request.args.get("file")

    if not filename:
        abort(400, description="Missing file parameter.")

    filepath = safe_document_path(filename)

    # Render Markdown directly in the browser
    if filepath.suffix.lower() == ".md":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return Response(
            content,
            mimetype="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'inline; filename="{filepath.name}"'
            }
        )

    # All other file types
    return send_file(
        filepath,
        as_attachment=False,
    )


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":

    print("=" * 70)
    print("Askly Document Server")
    print("=" * 70)
    print(f"Serving : {DOCUMENTS_FOLDER}")
    print(f"URL     : http://{HOST}:{PORT}")
    print("=" * 70)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
    )