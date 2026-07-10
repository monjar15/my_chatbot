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
# Custom Error Pages
# Flask's default abort() pages are plain, unbranded HTML — replace them
# with a simple friendly page so a broken/missing source link fails
# gracefully instead of showing a raw server error screen.
###############################################################################

def render_error_page(title: str, message: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title} — Askly</title>
        <style>
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                background-color: #c4c9ce;
                color: #2c3a3f;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }}
            .error-box {{
                background: #e7eaec;
                border: 1px solid #c2c9ce;
                border-radius: 12px;
                padding: 32px 40px;
                text-align: center;
                max-width: 420px;
            }}
            h1 {{ font-size: 20px; margin-bottom: 12px; }}
            p {{ font-size: 15px; color: #4b5358; }}
        </style>
    </head>
    <body>
        <div class="error-box">
            <h1>📄 {title}</h1>
            <p>{message}</p>
        </div>
    </body>
    </html>
    """

@app.errorhandler(400)
def handle_bad_request(e):
    return render_error_page(
        "Missing file",
        "No file was specified in the request."
    ), 400

@app.errorhandler(403)
def handle_forbidden(e):
    return render_error_page(
        "Access denied",
        "You don't have permission to view this file."
    ), 403

@app.errorhandler(404)
def handle_not_found(e):
    return render_error_page(
        "Source not found",
        "This document may have been moved, renamed, or deleted."
    ), 404

@app.errorhandler(500)
def handle_server_error(e):
    return render_error_page(
        "Something went wrong",
        "The document server ran into an unexpected error. Please try again."
    ), 500


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