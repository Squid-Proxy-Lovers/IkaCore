from smolagents import tool
import hashlib
import json
import os
import requests
import shlex
import time
from pathlib import Path

from .ctf_tools import get_ctf_environment

try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
except Exception:
    Chroma = None
    OpenAIEmbeddings = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except Exception:
    chromadb = None
    ChromaSettings = None

# Optional local embedding fallback
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

class _LocalEmbeddings:
    """
    Lightweight embeddings wrapper compatible with LangChain's Chroma integration.

    Uses sentence-transformers; set RAG_EMBEDDINGS_MODEL to override the model name.
    """
    def __init__(self, model_name: str | None = None):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; cannot use local embeddings")
        self.model = SentenceTransformer(model_name or os.getenv("RAG_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize to improve cosine similarity behavior
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def _ensure_openai_api_key() -> None:
    """
    If OPENAI_API_KEY is not set in the environment, attempt to load it from keys.cfg
    at the repository root.
    """
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        keys_file = Path(__file__).resolve().parent.parent.parent / "keys.cfg"
        if not keys_file.exists():
            return
        with open(keys_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENAI_API_KEY="):
                    value = line.split("=", 1)[1].strip()
                    if value:
                        os.environ["OPENAI_API_KEY"] = value
                    break
    except Exception:
        # Best-effort only; fall back to local embeddings if not available
        return

def _get_embeddings():
    """
    Returns an embedding function for Chroma. Prefers OpenAI if OPENAI_API_KEY is set
    and not disabled via RAG_USE_OPENAI=0. Otherwise, falls back to sentence-transformers.
    """
    use_openai = os.getenv("RAG_USE_OPENAI", "1") != "0"
    if use_openai and OpenAIEmbeddings is not None and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    if use_openai and OpenAIEmbeddings is not None:
        _ensure_openai_api_key()
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddings()
    # fallback to local embeddings
    return _LocalEmbeddings()


@tool
def list_directory(directory_path: str) -> str:
    """
    Lists the files and subdirectories within a specified directory path in the container.

    Args:
        directory_path (str): The path of the directory to inspect in the container (e.g., '.', '/tmp').
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(f"ls -la '{directory_path}'")

@tool
def get_file_size(file_path: str) -> str:
    """
    Gets the size of a file in bytes before reading it.
    Use this to check if a file is too large to read directly.

    Args:
        file_path (str): The full path to the file in the container.
    
    Returns:
        File size information and type.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    result = env.run_command_in_container(f"ls -lh '{file_path}'")
    result += "\n" + env.run_command_in_container(f"file '{file_path}'")
    result += "\n" + env.run_command_in_container(f"wc -l '{file_path}' 2>/dev/null || echo 'Cannot count lines (binary file?)'")
    return result

@tool
def read_file(file_path: str, section: int = None) -> str:
    """
    Reads and returns the content of a specified text file from the container, 
    limited to 300 lines maximum (1 section). If the file is longer, split into 300-line sections, 
    and require specifying which section to read. Only use this if you have a small amount of files use read_directory instead in most cases

    IMPORTANT: Never call more than 300 lines. Use get_file_size first for very large or binary files.

    Args:
        file_path (str): The full path to the file to be read in the container.
        section (int): Section of the file to read. Each section is 300 lines. If the file has multiple sections, agent must specify which section (starting from 1).
    Returns:
        If the file is <= 300 lines, returns its content. If more, returns only the requested section and info about total sections. If section is not specified, instruct agent to pick a section.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"

    line_count_result = env.run_command_in_container(f"wc -l '{file_path}'")
    try:
        line_count = int(line_count_result.strip().split()[0])
    except Exception:
        return f"Could not determine number of lines in file. wc -l output: {line_count_result}"

    lines_per_section = 300
    num_sections = (line_count + lines_per_section - 1) // lines_per_section

    if line_count <= lines_per_section:
        return env.run_command_in_container(f"cat '{file_path}'")

    if section is None or section < 1 or section > num_sections:
        return (
            f"File '{file_path}' has {line_count} lines and is divided into {num_sections} sections "
            f"(each section is 300 lines).\n"
            f"To read this file, please specify a section number between 1 and {num_sections} "
            f"using the 'section' argument."
        )

    start_line = 1 + (section - 1) * lines_per_section
    end_line = min(start_line + lines_per_section - 1, line_count)
    read_cmd = f"sed -n '{start_line},{end_line}p' '{file_path}'"
    content = env.run_command_in_container(read_cmd)
    return (
        f"Showing section {section}/{num_sections} (lines {start_line}-{end_line}) of '{file_path}':\n"
        f"{content}"
    )
def read_files_in_directory(directory_path: str) -> str:
    """
    Reads and returns the content of all text files in a specified directory from the container.
    Has a character limit to prevent overwhelming output. PREFER THIS OVER READING INDIVIDUAL FILES THIS SAVES TOOL CALLS
    Excludes common non-CTF files like CSS, lock files, and config files.

    Args:
        directory_path (str): The full path to the directory containing files to read.
    
    Returns:
        Combined content of all files in the directory, with file separators.
        If total content exceeds max_chars, returns partial content with a warning.
    """
    max_chars = 750000  # Character limit for output
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"

    # Check if directory exists
    check_dir = env.run_command_in_container(f"test -d '{directory_path}' && echo 'exists' || echo 'not_found'")
    if 'not_found' in check_dir:
        return f"Error: Directory '{directory_path}' does not exist"

    # Get list of files in directory
    list_files = env.run_command_in_container(f"find '{directory_path}' -type f")
    files = [f.strip() for f in list_files.strip().split('\n') if f.strip()]
    
    if not files:
        return f"No files found in directory '{directory_path}'"

    # Files and patterns to exclude (not relevant to CTF solving)
    exclude_patterns = [
        '.css', '.lock', '.entrypoint', '.bashrc', '.vimrc',
        '.profile', '.bash_logout', '.bash_history',
        '.gitignore', '.gitattributes', '.dockerignore',
        '.editorconfig', '.eslintrc', '.prettierrc',
        'package-lock.json', 'yarn.lock', 'Gemfile.lock',
        '.DS_Store', 'thumbs.db',
        '.min.js', '.min.css', 
        '.map', '__pycache__/', '.git/',
        '.env.example', 'LICENSE', 'COPYING'
    ]
    
    # Filter out excluded files
    def should_include(file_path):
        file_lower = file_path.lower()
        for pattern in exclude_patterns:
            if pattern.startswith('.') and not pattern.startswith('./'):
                # Extension check
                if file_lower.endswith(pattern):
                    return False
            elif pattern.endswith('/'):
                # Directory check
                if pattern in file_path:
                    return False
            else:
                # Filename check
                if pattern in file_lower:
                    return False
        return True
    
    filtered_files = [f for f in files if should_include(f)]
    
    if not filtered_files:
        return f"No relevant files found in directory '{directory_path}' after filtering"

    total_chars = 0
    output_parts = [f"Reading {len(filtered_files)} files from '{directory_path}' ({len(files) - len(filtered_files)} files excluded):\n"]
    files_read = 0
    truncated = False

    for file_path in filtered_files:
        if total_chars >= max_chars:
            truncated = True
            break
            
        # Read file content
        content = env.run_command_in_container(f"cat '{file_path}'")
        
        # Add file separator and content
        file_section = f"\n{'='*80}\nFile: {file_path}\n{'='*80}\n{content}\n"
        
        # Check if adding this file would exceed limit
        if total_chars + len(file_section) > max_chars:
            # Add partial content
            remaining_chars = max_chars - total_chars
            file_section = file_section[:remaining_chars]
            output_parts.append(file_section)
            truncated = True
            break
        
        output_parts.append(file_section)
        total_chars += len(file_section)
        files_read += 1

    result = ''.join(output_parts)
    
    if truncated:
        result += f"\n\n[WARNING: Output truncated at {max_chars} characters. Read {files_read}/{len(filtered_files)} files.]"
    
    return result
@tool
def write_file(file_path: str, content: str) -> str:
    """
    Writes the given content to a specified file in the container, overwriting it if it exists.

    Args:
        file_path (str): The path to the file to be written to in the container.
        content (str): The string content to write into the file.
    """
    import base64
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Use base64 encoding to safely handle any content including newlines and special characters
    # This avoids issues with echo and shell escaping
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
    command = f"echo '{encoded_content}' | base64 -d > '{file_path}'"
    return env.run_command_in_container(command)

@tool
def run_bash_command(command: str) -> str:
    """
    Executes a bash command in the container and returns its standard output.

    Args:
        command (str): The bash command to execute in the container (e.g., 'whoami').
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    return env.run_command_in_container(command)

@tool
def make_web_request(url: str) -> str:
    """
    Performs an HTTP GET request to a URL and returns the text content.

    Args:
        url (str): The full URL to fetch (e.g., 'https://www.google.com').
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    try:
        # response = requests.get(url)
        # response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        return env.run_command_in_container(f"curl '{url}'")
    except Exception as e:
        return f"An error occurred while making the web request: {str(e)}"

@tool
def unarchive_file(file_path: str, password: str = None) -> str:
    """
    Unarchives a file inside the container using the `file` command to detect archive type (rather than relying on file extensions).
    If the archive is password protected, use the password argument to provide the password.
    Supports zip, gzip, bzip2, tar, rar, and 7z formats.
    Args:
        file_path (str): The full path to the archive file in the container.
        password (str): The password to use for the archive.
    Returns:
        A list of the extracted files with full paths.
    """

    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"

    archive_dir = env.run_command_in_container(f"dirname '{file_path}'").strip()

    extracted_files = []
    all_extracted_paths = []

    password_keywords = (
        "zip archive",
        "rar archive",
        "7-zip archive",
    )

    quoted_password = shlex.quote(password) if password else ""

    def supports_password(file_type: str) -> bool:
        # Ensure case-insensitive matching
        file_type_lower = file_type.lower()
        return any(keyword in file_type_lower for keyword in password_keywords)

    def detect_type(path: str) -> str:
        return env.run_command_in_container(f"file -b '{path}'").strip().lower()

    def get_archive_contents(path: str, file_type: str, directory: str) -> list:
        """Get list of files that will be extracted from archive"""
        contents = []
        try:
            if "zip archive" in file_type:
                if password and supports_password(file_type):
                    # Use 7z for password-protected ZIPs since unzip -Z1 doesn't work with passwords
                    result = env.run_command_in_container(f"7z l -slt -p{quoted_password} '{path}' | grep '^Path = ' | cut -d' ' -f3-")
                    contents = [f"{directory}/{line.strip()}" for line in result.strip().split('\n') if line.strip() and line.strip() != path]
                else:
                    result = env.run_command_in_container(f"unzip -Z1 '{path}'")
                    contents = [f"{directory}/{line.strip()}" for line in result.strip().split('\n') if line.strip()]
            elif "gzip compressed data" in file_type:
                # Check if it's a tar.gz
                check = env.run_command_in_container(f"tar -tzf '{path}' 2>/dev/null || echo 'NOT_TAR'")
                if "NOT_TAR" not in check:
                    contents = [f"{directory}/{line.strip()}" for line in check.strip().split('\n') if line.strip()]
                else:
                    # Single file decompression
                    new_path = path.rstrip('.gz')
                    contents = [new_path]
            elif "bzip2 compressed data" in file_type:
                # Check if it's a tar.bz2
                check = env.run_command_in_container(f"tar -tjf '{path}' 2>/dev/null || echo 'NOT_TAR'")
                if "NOT_TAR" not in check:
                    contents = [f"{directory}/{line.strip()}" for line in check.strip().split('\n') if line.strip()]
                else:
                    # Single file decompression
                    new_path = path.rstrip('.bz2')
                    contents = [new_path]
            elif "tar archive" in file_type:
                result = env.run_command_in_container(f"tar -tf '{path}'")
                contents = [f"{directory}/{line.strip()}" for line in result.strip().split('\n') if line.strip()]
            elif "rar archive" in file_type:
                if password and supports_password(file_type):
                    result = env.run_command_in_container(f"unrar lb -p{quoted_password} '{path}'")
                else:
                    result = env.run_command_in_container(f"unrar lb '{path}'")
                contents = [f"{directory}/{line.strip()}" for line in result.strip().split('\n') if line.strip()]
            elif "7-zip archive" in file_type:
                if password and supports_password(file_type):
                    result = env.run_command_in_container(f"7z l -slt -p{quoted_password} '{path}' | grep '^Path = ' | cut -d' ' -f3-")
                else:
                    result = env.run_command_in_container(f"7z l -slt '{path}' | grep '^Path = ' | cut -d' ' -f3-")
                contents = [f"{directory}/{line.strip()}" for line in result.strip().split('\n') if line.strip() and line.strip() != path]
        except Exception:
            pass
        return contents

    def extract(path: str, directory: str):
        file_type = detect_type(path)  # Already lowercased by detect_type
        
        # Get contents before extraction
        contents = get_archive_contents(path, file_type, directory)
        all_extracted_paths.extend(contents)

        if "zip archive" in file_type:
            if password and supports_password(file_type):
                # Use 7z for password-protected ZIPs for consistency with listing and better AES support
                env.run_command_in_container(f"7z x -o'{directory}' -y -p{quoted_password} '{path}'")
            else:
                env.run_command_in_container(f"unzip -o '{path}' -d '{directory}'")
        elif "gzip compressed data" in file_type:
            new_path = path.rstrip('.gz')
            try:
                env.run_command_in_container(f"tar -tzf '{path}' > /dev/null 2>&1")
                env.run_command_in_container(f"tar -xzf '{path}' -C '{directory}'")
            except Exception:
                env.run_command_in_container(f"gunzip -c '{path}' > '{new_path}'")
                return new_path
        elif "bzip2 compressed data" in file_type:
            new_path = path.rstrip('.bz2')
            try:
                env.run_command_in_container(f"tar -tjf '{path}' > /dev/null 2>&1")
                env.run_command_in_container(f"tar -xjf '{path}' -C '{directory}'")
            except Exception:
                env.run_command_in_container(f"bunzip2 -c '{path}' > '{new_path}'")
                return new_path
        elif "tar archive" in file_type:
            env.run_command_in_container(f"tar -xf '{path}' -C '{directory}'")
        elif "rar archive" in file_type:
            if password and supports_password(file_type):
                env.run_command_in_container(f"unrar x -o+ -p{quoted_password} '{path}' '{directory}/'")
            else:
                env.run_command_in_container(f"unrar x -o+ '{path}' '{directory}/'")
        elif "7-zip archive" in file_type:
            if password and supports_password(file_type):
                env.run_command_in_container(f"7z x -o'{directory}' -y -p{quoted_password} '{path}'")
            else:
                env.run_command_in_container(f"7z x -o'{directory}' -y '{path}'")
        else:
            return None

        return None

    def recursive_extract(start_dir: str):
        files = env.run_command_in_container(f"find '{start_dir}' -type f").splitlines()
        for f in files:
            f = f.strip()
            if not f:
                continue
            file_type = detect_type(f)
            if any(keyword in file_type for keyword in [
                "zip archive", "gzip compressed data", "bzip2 compressed data",
                "tar archive", "rar archive", "7-zip archive"
            ]):
                new_file = extract(f, start_dir)
                extracted_files.append(f)
                if new_file and os.path.basename(new_file) != os.path.basename(f):
                    recursive_extract(start_dir)
        return

    recursive_extract(archive_dir)

    if not extracted_files:
        return f"No archives found or extracted for {file_path}"

    output = "=== Extracted Archives ===\n"
    output += "\n".join(extracted_files) + "\n\n"
    output += "=== All Extracted Files (Full Paths) ===\n"
    output += "\n".join(all_extracted_paths)
    
    return output

@tool
def read_rag_db(query: str) -> str:
    """
    Reads the RAG database and returns the most relevant information.

    Args:
        query (str): Natural language search query to retrieve relevant chunks.
    """
    try:
        # Lazy init if available
        if Chroma is None:
            return "RAG is not configured in this environment."
        try:
            embeddings = _get_embeddings()
        except Exception as ee:
            return f"RAG embedding not available: {ee}"
        RAG_DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db")
        collection_name = _get_active_collection(default="rag-chroma")
        vectorstore = Chroma(
          collection_name=collection_name,
          persist_directory=RAG_DB_DIR,
          embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        return "\n---\n".join([getattr(d, 'page_content', str(d)) for d in docs])
    except Exception as e:
        return f"Error reading from RAG DB: {e}"


@tool
def write_rag_db(content: str, metadata_json: str = "") -> str:
    """
    Writes a text chunk to the RAG database, with metadata as JSON.

    Copy the following metadata format and fill in the values as appropriate:
    EXAMPLE:
    {"agent":"IDA","challenge":"...","file":"...","line":"..."}

    Args:
        content: Text to index.
        metadata_json: JSON string with metadata regarding the agent that wrote the content 
    """
    try:
        if Chroma is None:
            return "RAG is not configured in this environment."
        try:
            embeddings = _get_embeddings()
        except Exception as ee:
            return f"RAG embedding not available: {ee}"
        RAG_DB_DIR = os.getenv("RAG_DB_DIR", "./rag_db")
        collection_name = _get_active_collection(default="rag-chroma")
        vectorstore = Chroma(
          collection_name=collection_name,
          persist_directory=RAG_DB_DIR,
          embedding_function=embeddings
        )
        metadata = json.loads(metadata_json) if metadata_json else {}
        doc_id = metadata.get("doc_id") or _compute_doc_id(metadata, content)
        
        # Check if document with this ID already exists
        try:
            existing = vectorstore.get(ids=[doc_id])
            if existing and existing.get("ids"):
                challenge = metadata.get("challenge", "unknown")
                return (
                    f"ERROR: Document with ID '{doc_id}' already exists in RAG database!\n"
                    f"Cannot overwrite existing entries. This prevents accidental data loss.\n"
                    f"\n"
                    f"To see existing entries, use: list_rag_db(challenge='{challenge}')\n"
                    f"To search for content, use: query_rag_db(query='your search')\n"
                    f"\n"
                    f"If you want to add NEW information, modify your metadata to generate a different doc_id.\n"
                    f"Current metadata: {metadata_json}"
                )
        except Exception:
            # If get() fails, assume document doesn't exist and proceed
            pass
        
        ids = vectorstore.add_texts(texts=[content], metadatas=[metadata], ids=[doc_id])
        return f"Indexed 1 document with id(s): {ids}"
    except Exception as e:
        return f"Error writing to RAG DB: {e}"

# ------------------------
# Advanced RAG management
# ------------------------

def _rag_dir() -> str:
    return os.getenv("RAG_DB_DIR", "./rag_db")

def _collection_marker_path() -> Path:
    return Path(_rag_dir()) / ".active_collection"

def _get_active_collection(default: str = "rag-chroma") -> str:
    try:
        p = _collection_marker_path()
        if p.exists():
            return p.read_text().strip() or default
        return default
    except Exception:
        return default

def _set_active_collection(name: str) -> None:
    p = _collection_marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(name)

def _get_chromadb_collection(collection_name: str):
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    path = _rag_dir()
    try:
        client = chromadb.PersistentClient(path=path)  # chromadb>=0.5
    except Exception:
        # Fallback: older API with Settings
        client = chromadb.Client(ChromaSettings(persist_directory=path))  # type: ignore
    return client.get_or_create_collection(name=collection_name)

def _compute_doc_id(metadata: dict, content: str) -> str:
    key_fields = [
        metadata.get("challenge", ""),
        metadata.get("binary", ""),
        metadata.get("type", ""),
        metadata.get("label", ""),
        metadata.get("address", ""),
        metadata.get("file", ""),
        str(metadata.get("stage", "")),
    ]
    h = hashlib.sha1("|".join(key_fields).encode("utf-8"))
    return h.hexdigest()

@tool
def set_rag_collection(name: str) -> str:
    """
    Sets the active RAG collection name for subsequent operations.

    Args:
        name (str): Collection name to select, e.g., "rev-<challenge_slug>".
    """
    try:
        _set_active_collection(name)
        return f"Active RAG collection set to: {name}"
    except Exception as e:
        return f"Error setting collection: {e}"

@tool
def get_rag_collection() -> str:
    """
    Returns the current active RAG collection name.
    """
    return _get_active_collection()

@tool
def search_rag_db(query: str, where_json: str = "", k: int = 8, collection: str = "") -> str:
    """
    Hybrid search with optional metadata filters. Returns JSONL of {id, metadata, snippet}.

    Args:
        query (str): Natural language query used for vector retrieval.
        where_json (str): JSON dict filter on metadata (e.g., '{"challenge":"ctf-1"}').
        k (int): Number of results to return.
        collection (str): Override collection name; defaults to the active collection.
    """
    try:
        if Chroma is None:
            return "RAG is not configured in this environment."
        try:
            embeddings = _get_embeddings()
        except Exception as ee:
            return f"RAG embedding not available: {ee}"
        RAG_DB_DIR = _rag_dir()
        collection_name = collection or _get_active_collection("rag-chroma")
        vectorstore = Chroma(
          collection_name=collection_name,
          persist_directory=RAG_DB_DIR,
          embedding_function=embeddings
        )
        flt = json.loads(where_json) if where_json else None
        retriever = vectorstore.as_retriever(search_kwargs={"k": int(k), "filter": flt} if flt else {"k": int(k)})
        docs = retriever.invoke(query)
        lines = []
        for d in docs:
            meta = getattr(d, 'metadata', {}) or {}
            doc_id = meta.get("doc_id") or meta.get("id") or ""
            snippet = (getattr(d, 'page_content', str(d)) or "").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(json.dumps({"id": doc_id, "metadata": meta, "snippet": snippet}))
        return "\n".join(lines) if lines else "[]"
    except Exception as e:
        return f"Error searching RAG DB: {e}"

@tool
def update_rag_db(doc_id: str, new_content: str = "", new_metadata_json: str = "", collection: str = "") -> str:
    """
    Updates an existing doc by id. If content or metadata omitted, preserves current value.

    Args:
        doc_id (str): The document id to update.
        new_content (str): New full content for the document. Leave empty to keep existing.
        new_metadata_json (str): JSON dict of metadata fields to merge into existing metadata.
        collection (str): Override collection name; defaults to the active collection.
    """
    try:
        coll = _get_chromadb_collection(collection or _get_active_collection("rag-chroma"))
        cur = coll.get(ids=[doc_id])
        if not cur or not cur.get("ids"):
            return f"Not found: {doc_id}"
        doc = (cur.get("documents") or [""])[0]
        meta = (cur.get("metadatas") or [{}])[0]
        if new_content:
            doc = new_content
        if new_metadata_json:
            try:
                meta_update = json.loads(new_metadata_json)
                meta.update(meta_update)
            except Exception:
                return "Invalid JSON for new_metadata_json"
        coll.update(ids=[doc_id], documents=[doc], metadatas=[meta])
        return f"Updated: {doc_id}"
    except Exception as e:
        return f"Error updating RAG DB: {e}"

@tool
def delete_rag_db(doc_ids_json: str = "", where_json: str = "", collection: str = "") -> str:
    """
    Deletes documents by ids or metadata filter.

    Args:
        doc_ids_json (str): JSON list of ids to delete (e.g., '["id1","id2"]').
        where_json (str): JSON dict metadata filter for bulk delete (e.g., '{"challenge":"ctf-1"}').
        collection (str): Override collection name; defaults to the active collection.
    """
    try:
        coll = _get_chromadb_collection(collection or _get_active_collection("rag-chroma"))
        ids = json.loads(doc_ids_json) if doc_ids_json else None
        where = json.loads(where_json) if where_json else None
        if ids:
            coll.delete(ids=ids)
            return f"Deleted {len(ids)} by id"
        if where:
            coll.delete(where=where)
            return "Deleted by filter"
        return "No ids or filter provided"
    except Exception as e:
        return f"Error deleting from RAG DB: {e}"

@tool
def list_rag_db(where_json: str = "", limit: int = 100, collection: str = "") -> str:
    """
    Lists documents (id, metadata, snippet) matching optional filter.

    Args:
        where_json (str): JSON dict metadata filter (e.g., '{"type":"func"}').
        limit (int): Maximum number of documents to return.
        collection (str): Override collection name; defaults to the active collection.
    """
    try:
        coll = _get_chromadb_collection(collection or _get_active_collection("rag-chroma"))
        where = None
        if where_json and where_json.strip():
            try:
                where = json.loads(where_json)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in where_json parameter: {where_json}"
        resp = coll.get(where=where, limit=int(limit))
        ids = resp.get("ids") or []
        docs = resp.get("documents") or []
        metas = resp.get("metadatas") or []
        lines = []
        for i, doc_id in enumerate(ids):
            snippet = (docs[i] or "").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(json.dumps({"id": doc_id, "metadata": metas[i] if i < len(metas) else {}, "snippet": snippet}))
        return "\n".join(lines) if lines else "[]"
    except Exception as e:
        return f"Error listing RAG DB: {e}"

@tool
def get_rag_doc(doc_id: str, collection: str = "") -> str:
    """
    Gets a single document by id, returning full document and metadata as JSON.

    Args:
        doc_id (str): The document id to fetch.
        collection (str): Override collection name; defaults to the active collection.
    """
    try:
        coll = _get_chromadb_collection(collection or _get_active_collection("rag-chroma"))
        resp = coll.get(ids=[doc_id])
        if not resp or not resp.get("ids"):
            return f"Not found: {doc_id}"
        result = {
            "id": (resp.get("ids") or [""])[0],
            "metadata": (resp.get("metadatas") or [{}])[0],
            "document": (resp.get("documents") or [""])[0],
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error getting RAG doc: {e}"

# ------------------------
# Simple JSON blackboard
# ------------------------

def _blackboard_path() -> Path:
    root = Path("./artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / "blackboard.json"

@tool
def blackboard_write(key: str, json_value: str) -> str:
    """
    Writes a JSON value at key into a shared blackboard file.

    Args:
        key (str): The key name to write under.
        json_value (str): JSON-encoded value to write.
    """
    p = _blackboard_path()
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text() or "{}")
        except Exception:
            data = {}
    try:
        value = json.loads(json_value)
    except Exception:
        return "Invalid JSON in json_value"
    data[key] = value
    p.write_text(json.dumps(data))
    return f"blackboard[{key}] written"

@tool
def blackboard_read(key: str) -> str:
    """
    Reads a value from the shared blackboard file by key.

    Args:
        key (str): The key name to read.
    """
    p = _blackboard_path()
    if not p.exists():
        return "{}"
    try:
        data = json.loads(p.read_text() or "{}")
        return json.dumps(data.get(key)) if key in data else "null"
    except Exception as e:
        return f"Error reading blackboard: {e}"

@tool
def blackboard_list(prefix: str = "") -> str:
    """
    Lists keys in the blackboard, optionally filtering by prefix.

    Args:
        prefix (str): Optional prefix to filter keys by.
    """
    p = _blackboard_path()
    if not p.exists():
        return "[]"
    try:
        data = json.loads(p.read_text() or "{}")
        keys = sorted([k for k in data.keys() if not prefix or k.startswith(prefix)])
        return json.dumps(keys)
    except Exception as e:
        return f"Error listing blackboard: {e}"

@tool
def run_solve_script(script_path: str) -> str:
    """
    Runs a Python/Sage script in the container.

    Args:
        script_path (str): The path to the Python script to run in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"  
    if script_path.endswith(".py"):
        return env.run_command_in_container(f"python3 '{script_path}'")
    elif script_path.endswith(".sage"):
        return env.run_command_in_container(f"sage '{script_path}'")
    else:
        return "Error: Unsupported script type. Only .py and .sage files are supported."
    
@tool
def list_workspace_files() -> str:
    """
    Lists all files in the workspace directory, including generated files and analysis results.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    files = env.list_workspace_files()
    if files:
        return "Workspace Files:\n" + "\n".join(f"- {file}" for file in files)
    else:
        return "No files found in workspace"

@tool
def run_container_command(command: str) -> str:
    """
    Executes a command inside the CTF Docker container with access to analysis tools.
    
    Args:
        command (str): The command to execute in the container.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    return env.run_command_in_container(command, "/home/ctfplayer/")

@tool
def run_local_command(command: str) -> str:
    """
    Executes a command locally in the workspace directory.
    
    Args:
        command (str): The command to execute locally.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    return env.run_command_local(command)


@tool
def get_remote_info() -> str:
    """
    Retrieves the IP address and open ports of the challenge environment.
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    info = env.challenge_ip

    if info is None or info == "":
        return "Error: Challenge environment IP not found Check description for any remote info "
    info += "\n"
    for port, mapped_port in env.challenge_service_ports.items():
        info += f"Port {port}\n"
    return info
# ============================================================================
# SMART FILE READING - Minimize token usage
# ============================================================================

@tool
def read_file_lines(file_path: str, start_line: int = 1, end_line: int = None) -> str:
    """
    Read specific lines from a file. Much more efficient than reading entire sections.
    
    Args:
        file_path: Path to file in container
        start_line: First line to read (1-indexed, defaults to 1)
        end_line: Last line to read (inclusive). If None, reads to end of file
    
    Returns:
        Requested lines with line numbers prepended
    
    Example:
        read_file_lines("main.c", 45, 60)
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Validate start_line
    if start_line < 1:
        return "Error: start_line must be >= 1"
    
    # Build sed command for line range
    if end_line is None:
        # Read from start_line to end of file
        sed_cmd = f"sed -n '{start_line},$p' '{file_path}'"
    else:
        if end_line < start_line:
            return f"Error: end_line ({end_line}) must be >= start_line ({start_line})"
        sed_cmd = f"sed -n '{start_line},{end_line}p' '{file_path}'"
    
    # Add line numbers for reference
    cmd = f"{sed_cmd} | nl -ba -v {start_line}"
    content = env.run_command_in_container(cmd)
    
    # Get total line count for context
    line_count_result = env.run_command_in_container(f"wc -l '{file_path}'")
    try:
        total_lines = int(line_count_result.strip().split()[0])
    except:
        total_lines = "unknown"
    
    return f"File: {file_path} (showing lines {start_line}-{end_line or 'EOF'}, total: {total_lines})\n{content}"


@tool
def read_file_ranges(ranges_json: str) -> str:
    """
    Read multiple line ranges from multiple files in ONE operation. Highly efficient.
    
    Args:
        ranges_json: JSON array of [file_path, start_line, end_line] tuples. Example: '[["main.c", 1, 50], ["utils.c", 100, 150]]'
    
    Returns:
        All requested content with clear file/line separators
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    try:
        ranges = json.loads(ranges_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    
    results = []
    for item in ranges:
        if len(item) < 2 or len(item) > 3:
            return f"Error: Each range must be [file_path, start_line] or [file_path, start_line, end_line]"
        
        file_path = item[0]
        start_line = item[1]
        end_line = item[2] if len(item) == 3 else None
        
        # Use the existing read_file_lines logic
        content = read_file_lines(file_path, start_line, end_line)
        results.append(content)
        results.append("=" * 80)
    
    return "\n".join(results)


@tool
def get_file_outline(file_path: str, pattern: str = r"^(def |class |function |struct |void |int |char )") -> str:
    """
    Get a structural outline of a file (function/class definitions, etc.) without reading full content.
    Uses minimal tokens to understand file structure.
    
    Args:
        file_path: Path to file in container
        pattern: Regex pattern to match important lines (defaults to common function/class declarations)
    
    Returns:
        Line numbers and matching lines showing file structure
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Use grep to find structural elements with line numbers
    cmd = f"grep -n -E '{pattern}' '{file_path}' || echo 'No matches found'"
    result = env.run_command_in_container(cmd)
    
    # Get file info
    info_cmd = f"wc -l '{file_path}' && file '{file_path}'"
    info = env.run_command_in_container(info_cmd)
    
    return f"File outline for: {file_path}\n{info}\n\nStructural elements:\n{result}"


@tool
def search_in_file(file_path: str, search_term: str, context_lines: int = 2) -> str:
    """
    Search for a term in a file and show surrounding context. Efficient for finding specific code.
    
    Args:
        file_path: Path to file in container
        search_term: String or regex to search for
        context_lines: Number of lines of context before/after each match (defaults to 2)
    
    Returns:
        Matching lines with line numbers and context
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Use grep with context and line numbers
    cmd = f"grep -n -C {context_lines} -E '{search_term}' '{file_path}' || echo 'No matches found'"
    result = env.run_command_in_container(cmd)
    
    return f"Search results for '{search_term}' in {file_path}:\n{result}"


# ============================================================================
# SMART FILE WRITING - Line-based patches with conflict detection
# ============================================================================

@tool
def get_file_hash(file_path: str, start_line: int = 1, end_line: int = None) -> str:
    """
    Get hash of file content for conflict detection. Used before editing.
    
    Args:
        file_path: Path to file in container
        start_line: First line of range to hash (defaults to 1)
        end_line: Last line of range to hash. If None, hashes entire file from start_line
    
    Returns:
        JSON with hash, line count, and line range
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Get the content we'll hash
    if end_line is None:
        sed_cmd = f"sed -n '{start_line},$p' '{file_path}'"
    else:
        sed_cmd = f"sed -n '{start_line},{end_line}p' '{file_path}'"
    
    # Calculate hash of the content
    hash_cmd = f"{sed_cmd} | sha256sum | cut -d' ' -f1"
    file_hash = env.run_command_in_container(hash_cmd).strip()
    
    # Get total line count
    line_count_result = env.run_command_in_container(f"wc -l '{file_path}'")
    try:
        total_lines = int(line_count_result.strip().split()[0])
    except:
        total_lines = 0
    
    result = {
        "file_path": file_path,
        "hash": file_hash,
        "total_lines": total_lines,
        "hashed_range": f"{start_line}-{end_line or 'EOF'}"
    }
    
    return json.dumps(result, indent=2)

@tool
def patch_file_lines(file_path: str, start_line: int, end_line: int, 
                     new_content: str) -> str:
    """
    Replace a range of lines with new content.
    
    Args:
        file_path: Path to file in container
        start_line: First line to replace (1-indexed)
        end_line: Last line to replace (inclusive)
        new_content: New content to insert (will replace lines start_line through end_line)
    
    Returns:
        Success message or error
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Create temporary file with new content
    import base64
    import hashlib
    temp_file = f"/tmp/patch_{hashlib.md5(file_path.encode()).hexdigest()}.tmp"
    encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
    create_temp = f"echo '{encoded_content}' | base64 -d > '{temp_file}'"
    env.run_command_in_container(create_temp)
    
    # Apply the patch: lines 1 to start_line-1, then new content, then end_line+1 to EOF
    cmd = f"""
    cp '{file_path}' '{file_path}.backup' && \
    sed -n '1,$(({{start_line}}-1))p' '{file_path}.backup' > '{file_path}.new' && \
    cat '{temp_file}' >> '{file_path}.new' && \
    sed -n '$(({{end_line}}+1)),$p' '{file_path}.backup' >> '{file_path}.new' && \
    mv '{file_path}.new' '{file_path}' && \
    rm '{temp_file}' '{file_path}.backup'
    """.replace("{start_line}", str(start_line)).replace("{end_line}", str(end_line))
    
    result = env.run_command_in_container(cmd)
    
    # Verify the operation succeeded
    verify = env.run_command_in_container(f"test -f '{file_path}' && echo 'OK' || echo 'FAILED'")
    
    if "OK" in verify:
        new_count = env.run_command_in_container(f"wc -l '{file_path}'")
        return f"Successfully patched {file_path} (lines {start_line}-{end_line})\n{new_count}"
    else:
        return f"Error: Patch operation failed: {result}"


@tool
def insert_lines(file_path: str, after_line: int, new_content: str) -> str:
    """
    Insert new lines after a specific line number.
    
    Args:
        file_path: Path to file in container
        after_line: Insert new content after this line (0 = beginning of file)
        new_content: Content to insert
    
    Returns:
        Success message or error
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Create temp file with new content
    import base64
    import hashlib
    temp_file = f"/tmp/insert_{hashlib.md5(file_path.encode()).hexdigest()}.tmp"
    encoded_content = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
    create_temp = f"echo '{encoded_content}' | base64 -d > '{temp_file}'"
    env.run_command_in_container(create_temp)
    
    if after_line == 0:
        # Insert at beginning
        cmd = f"""
        cat '{temp_file}' '{file_path}' > '{file_path}.new' && \
        mv '{file_path}.new' '{file_path}' && \
        rm '{temp_file}'
        """
    else:
        # Insert after specific line
        cmd = f"""
        sed -n '1,{after_line}p' '{file_path}' > '{file_path}.new' && \
        cat '{temp_file}' >> '{file_path}.new' && \
        sed -n '{after_line + 1},$p' '{file_path}' >> '{file_path}.new' && \
        mv '{file_path}.new' '{file_path}' && \
        rm '{temp_file}'
        """
    
    result = env.run_command_in_container(cmd)
    new_count = env.run_command_in_container(f"wc -l '{file_path}'")
    return f"Successfully inserted content after line {after_line}\n{new_count}"


@tool
def delete_lines(file_path: str, start_line: int, end_line: int) -> str:
    """
    Delete a range of lines from a file.
    
    Args:
        file_path: Path to file in container
        start_line: First line to delete (1-indexed)
        end_line: Last line to delete (inclusive)
    
    Returns:
        Success message or error
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Use sed to delete lines
    cmd = f"sed -i '{start_line},{end_line}d' '{file_path}'"
    result = env.run_command_in_container(cmd)
    
    new_count = env.run_command_in_container(f"wc -l '{file_path}'")
    return f"Successfully deleted lines {start_line}-{end_line}\n{new_count}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@tool
def get_file_stats(file_path: str) -> str:
    """
    Get comprehensive file statistics without reading content. Very token-efficient.
    
    Args:
        file_path: Path to file in container
    
    Returns:
        JSON with size, line count, file type, permissions, and hash
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Gather all info in one go
    cmd = f"""
    echo "{{" && \
    echo "  \\"path\\": \\"{file_path}\\"," && \
    echo -n "  \\"size_bytes\\": " && stat -c %s '{file_path}' && echo "," && \
    echo -n "  \\"permissions\\": \\"" && stat -c %A '{file_path}' && echo "\\"," && \
    echo -n "  \\"line_count\\": " && wc -l '{file_path}' | cut -d' ' -f1 && echo "," && \
    echo -n "  \\"file_type\\": \\"" && file -b '{file_path}' | sed 's/"/\\\\"/g' && echo "\\"," && \
    echo -n "  \\"sha256\\": \\"" && sha256sum '{file_path}' | cut -d' ' -f1 && echo "\\"" && \
    echo "}}"
    """
    
    return env.run_command_in_container(cmd)


@tool  
def create_file_with_content(file_path: str, content: str) -> str:
    """
    Create a new file with initial content. Fails if file exists (safety).
    
    Args:
        file_path: Path for new file in container
        content: Initial content to write to the file
    
    Returns:
        Success message with file stats or error if file already exists
    """
    env = get_ctf_environment()
    if not env:
        return "Error: No CTF environment is currently active"
    
    # Check if file exists
    check = env.run_command_in_container(f"test -f '{file_path}' && echo 'exists' || echo 'new'")
    if 'exists' in check:
        return f"Error: File '{file_path}' already exists. Use patch_file_lines to modify it."
    
    # Create file
    import base64
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
    cmd = f"echo '{encoded_content}' | base64 -d > '{file_path}'"
    env.run_command_in_container(cmd)
    
    return f"Created file: {file_path}\n{get_file_stats(file_path)}"