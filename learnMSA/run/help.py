import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List


# Documentation directory (inside the learnMSA package)
HELP_DIR = Path(__file__).parent.parent / "docs"


def format_description(description: str) -> str:
    """Remove indentation from description lines."""
    lines = description.split('\n')
    # Remove leading whitespace from each line
    formatted_lines = [line.lstrip() for line in lines]
    return '\n'.join(formatted_lines)


def is_perfect_match(search_term: str, arg_name: str) -> bool:
    """Check if search_term is a perfect match within arg_name.

    A perfect match is when the search term matches:
    - The entire string, or
    - A substring surrounded by whitespace or start/end of string
    """
    search_lower = search_term.lower()
    name_lower = arg_name.lower()

    # Check if it's an exact match
    if search_lower == name_lower:
        return True

    # Split by common delimiters and check each token
    # Common delimiters in argument names: space, /, comma
    import re
    tokens = re.split(r'[\s/,]+', name_lower)

    return search_lower in tokens


@dataclass
class ArgumentDescription:
    """Represents a parsed argument from RST documentation."""
    name: str  # Full argument string, e.g., "-i / --in_file"
    description: str  # Full description text


def handle_help_command() -> bool:
    """Custom handling for 'help' as a pseudo-subcommand. Returns True if help
    was handled and the program should exit.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        if len(sys.argv) > 2:
            argument = sys.argv[2]
            show_detailed_help(argument)
        else:
            print("Usage: learnMSA help [argument]\n")
        return True
    return False


def show_detailed_help(argument: str) -> None:
    """Search and display help for arguments matching the given string.

    Searches all RST files for arguments containing the argument string as a substring.
    If a perfect match is found, only show perfect matches.
    """
    argument_lower = argument.lower()
    matching_args = []
    perfect_matches = []

    for rst_file in sorted(HELP_DIR.glob("*.rst")):
        if rst_file.stem in ["index", "getting_started", "api"]:
            continue

        with open(rst_file, "r", encoding="utf-8") as f:
            content = f.read()

        arguments = parse_rst_arguments(content)
        for arg in arguments:
            if argument_lower in arg.name.lower():
                match = (rst_file.stem, arg)
                matching_args.append(match)

                # Check if this is a perfect match
                if is_perfect_match(argument, arg.name):
                    perfect_matches.append(match)

    # If perfect matches exist, only show those
    results_to_show = perfect_matches if perfect_matches else matching_args

    if results_to_show:
        print(f"\nArguments matching '{argument}':")
        print("=" * (len(argument) + 20))
        print()

        for file_stem, arg in results_to_show:
            print(f"From: {file_stem.replace('_', ' ').title()}")
            print()
            print(f"{arg.name}")
            print(format_description(arg.description))
            print()
    else:
        print(f"No arguments found matching '{argument}'.\n")


def parse_rst_arguments(content: str) -> List[ArgumentDescription]:
    """Parse RST content and extract the Arguments section if present.

    Args:
        content: The full RST file content

    Returns:
        List of ArgumentDescription objects containing parsed arguments.
        Empty list if no Arguments section is found.
    """
    lines = content.split('\n')
    arguments = []

    # Find the Arguments section
    args_section_start = None
    args_section_end = None

    for i, line in enumerate(lines):
        if line.strip() == "Arguments" and i + 1 < len(lines):
            # Check if the next line is a section underline
            next_line = lines[i + 1].strip()
            if next_line and len(set(next_line)) == 1 and next_line[0] == '-':
                args_section_start = i
                break

    if args_section_start is None:
        return arguments

    # Find where the Arguments section ends (next section or end of file)
    for i in range(args_section_start + 2, len(lines)):
        # Check for next section (line followed by === or --- underline)
        if i + 1 < len(lines) and lines[i].strip() and lines[i + 1].strip():
            underline_chars = set(lines[i + 1].strip())
            if (len(underline_chars) == 1 and
                underline_chars.pop() in ['=', '-', '^', '~'] and
                len(lines[i + 1].strip()) >= len(lines[i].strip()) // 2):
                args_section_end = i
                break

    if args_section_end is None:
        args_section_end = len(lines)

    # Parse individual arguments
    i = args_section_start + 2  # Skip "Arguments" and underline
    # Regex to match a full-line argument declaration like: ``-i / --in_file`` *INPUT*
    arg_decl_re = re.compile(r"^\s*``([^`]+)``(?:\s+\*[^\*]+\*\s*)?$")
    while i < args_section_end:
        line = lines[i]

        # Treat as argument only when the whole line is an argument declaration
        m = arg_decl_re.match(line)
        if m:
            arg_name_clean = m.group(1).strip()

            i += 1

            # Collect description lines (indented lines that follow)
            description_lines = []
            while i < args_section_end:
                if not lines[i].strip():
                    # Empty line - end of this argument's description
                    i += 1
                    break
                # If the next non-empty line also matches an argument declaration,
                # stop collecting description (it's the next argument)
                if arg_decl_re.match(lines[i]):
                    break
                else:
                    description_lines.append(lines[i])
                    i += 1

            # Join description lines and clean up
            description = '\n'.join(description_lines).strip()

            # Create ArgumentDescription object
            arguments.append(ArgumentDescription(
                name=arg_name_clean,
                description=description
            ))
        else:
            i += 1

    return arguments
